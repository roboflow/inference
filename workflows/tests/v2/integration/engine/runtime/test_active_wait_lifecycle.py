"""Actual-engine CPU regressions for completion and owned-thread lifetime."""
import threading
import unittest
from unittest.mock import patch
from types import SimpleNamespace

LIMIT = 5.0

from roboflow_workflows.v2.data.kinds import FLOAT_KIND
from roboflow_workflows.v2.engine.active import runtime
from roboflow_workflows.v2.engine.active.pipeline import PipelinedDriver
from roboflow_workflows.v2.engine.errors import ActiveRunError, ContractError
from roboflow_workflows.v2.engine.pipelining.options import PipelineOptions
from roboflow_workflows.v2.engine.catalogue import Catalogue
from roboflow_workflows.v2.engine.compilation import compile_workflow
from roboflow_workflows.v2.engine.sources import Emission, Source, SourceOutput

class One(Source):
    type = 'review/one@v1'
    outputs = {'value': SourceOutput(FLOAT_KIND)}

    def __init__(self, *, opened: threading.Event):
        self.opened = opened
        self.sent = False

    def open(self):
        if not self.opened.wait(LIMIT):
            raise RuntimeError('test start gate timed out')

    def read(self):
        if self.sent:
            return None
        self.sent = True
        return Emission({'value': 1.0})

    def close(self):
        pass


class FinalLocal:
    """A CPU resource whose cleanup is observable when its thread frame drops."""
    def __init__(self, cleaned):
        self.cleaned = cleaned

    def __del__(self):
        self.cleaned.set()


class ActiveWaitLifecycle(unittest.TestCase):
    def session(self, opened):
        definition = dict(version='2.0', inputs=[], steps=[],
            sources=[dict(type=One.type, name='one')],
            outputs=[dict(type='OutputGroup', name='out', anchor='$sources.one.value',
                outputs=[dict(type='JsonField', name='value', selector='$sources.one.value')])])
        plan = compile_workflow(definition, catalogue=Catalogue([], sources=[One]))
        return plan.create_session(resources={'opened': opened})

    def threads(self, run):
        controller = getattr(run._driver, '_processor', None)
        if controller is None:
            controller = run._driver._dispatcher
        return [*run._readers.values(), controller]

    def exercise(self, boundary, *, failed=False):
        opened, entered, release, cleaned = (threading.Event() for _ in range(4))
        delivered = []
        session = self.session(opened)
        driver_type = PipelinedDriver if boundary == 'dispatcher' else runtime._SerialDriver
        method = {'reader': 'reader_finished', 'processor': '_process', 'dispatcher': '_dispatch'}[boundary]
        original = getattr(driver_type, method)

        def delayed(driver, *args):
            # The original queue notification/conclusion and real execution run.
            # Only final frame unwinding is held; _done and wait are not mocked.
            final_local = FinalLocal(cleaned)
            original(driver, *args)
            entered.set()
            if not release.wait(LIMIT):
                raise RuntimeError('test unwind gate timed out')
            # Explicit use keeps the local alive through the controlled hold.
            self.assertIs(final_local.cleaned, cleaned)

        run = None
        def handler(result):
            delivered.append(result.rows()[0]['value'])
            if failed:
                raise ValueError('controlled handler failure')

        options = {'pipeline': PipelineOptions(max_in_flight=2)} if boundary == 'dispatcher' else {}
        with patch.object(driver_type, method, delayed):
            try:
                run = session.start(handlers={'out': handler}, **options)
                opened.set()
                self.assertTrue(entered.wait(LIMIT), 'real engine never reached unwind boundary')
                thread = run._readers['one'] if boundary == 'reader' else self.threads(run)[-1]
                self.assertTrue(thread.is_alive())
                self.assertFalse(cleaned.is_set())
                observed = run.wait(timeout=0.1)
                self.assertEqual(delivered, [1.0])
                self.assertFalse(observed, 'wait returned before its owned thread frame finished')
                release.set()
                if failed:
                    with self.assertRaisesRegex(ActiveRunError, 'controlled handler failure'):
                        run.wait(timeout=LIMIT)
                else:
                    self.assertTrue(run.wait(timeout=LIMIT))
                    self.assertTrue(run.wait(timeout=0), 'repeat wait must remain complete')
                # A stored failure traceback can retain completed frames; join
                # guarantees thread exit, not destruction of referenced objects.
                if not failed:
                    self.assertTrue(cleaned.is_set())
                self.assertFalse(thread.is_alive())
            finally:
                opened.set()
                release.set()
                if run is not None:
                    run.stop()
                    for thread in self.threads(run):
                        if thread.ident is not None:
                            thread.join(LIMIT)
                            self.assertFalse(thread.is_alive(), 'test left an owned thread running')
                session.close()

    def test_wait_includes_reader_frame_unwind(self):
        self.exercise('reader')

    def test_wait_includes_processor_frame_unwind(self):
        self.exercise('processor')

    def test_wait_includes_pipelined_dispatcher_unwind(self):
        self.exercise('dispatcher')

    def test_failure_is_raised_after_processor_cleanup(self):
        self.exercise('processor', failed=True)

    def test_owned_handler_cannot_wait_for_itself(self):
        for pipeline in (None, PipelineOptions(max_in_flight=2)):
            with self.subTest(pipeline=pipeline):
                opened = threading.Event()
                session = self.session(opened)
                caught = []
                run = None
                def handler(result):
                    try:
                        run.wait(timeout=0)
                    except ContractError as error:
                        caught.append(str(error))
                try:
                    run = session.start(handlers={'out': handler}, pipeline=pipeline)
                    opened.set()
                    self.assertTrue(run.wait(timeout=LIMIT))
                    self.assertEqual(len(caught), 1)
                    self.assertIn('wait for itself', caught[0])
                finally:
                    opened.set()
                    if run is not None:
                        run.stop()
                        for thread in self.threads(run):
                            if thread.ident is not None:
                                thread.join(LIMIT)
                    session.close()

    def test_one_deadline_covers_event_readers_and_driver(self):
        # Isolate deadline arithmetic from scheduling; exercise the actual wait
        # method with completed barriers and a clock advanced at each boundary.
        calls = []
        class Joined:
            ident = 1
            def join(self, timeout):
                calls.append(('reader', timeout))
            def is_alive(self):
                return False
        subject = SimpleNamespace(
            _owned=SimpleNamespace(reject_wait=lambda action: None),
            _done=SimpleNamespace(wait=lambda timeout: calls.append(('event', timeout)) or True),
            _readers={'one': Joined(), 'two': Joined(), 'unstarted': threading.Thread()},
            _driver=SimpleNamespace(join=lambda timeout: calls.append(('driver', timeout)) or True),
            _failure=None)
        with patch.object(runtime, 'time', SimpleNamespace(monotonic=iter([10, 10.1, 10.3, 10.7, 11.2]).__next__)):
            self.assertTrue(runtime.ActiveRun.wait(subject, timeout=1))
        self.assertEqual([name for name, _ in calls], ['event', 'reader', 'reader', 'driver'])
        for (_, actual), expected in zip(calls, [.9, .7, .3, 0]):
            self.assertAlmostEqual(actual, expected)

    def test_drivers_allow_controller_never_started(self):
        for driver_type, attribute in ((runtime._SerialDriver, '_processor'), (PipelinedDriver, '_dispatcher')):
            with self.subTest(driver=driver_type.__name__):
                driver = object.__new__(driver_type)
                setattr(driver, attribute, threading.Thread())
                self.assertTrue(driver.join(timeout=0))


if __name__ == '__main__':
    unittest.main()
