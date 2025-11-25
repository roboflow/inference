import unittest

from lidra.test.util import run_unittest
from lidra.data.processor import Processor
from lidra.test.util import temporary_directory


class UnitTests(unittest.TestCase):
    def test_processor(self):
        def example_process_fn(item):
            print(f"processing item: {item}")
            if item % 5 == 0:
                raise ValueError("Oh no! Something crashed!")

        with temporary_directory() as temp_dir:
            data = list(range(1000))
            processor = Processor(
                path=temp_dir,
                data=data,
                process_fn=example_process_fn,
                timeout=1,
            )

            # check initial unprocessed status
            status = processor.status()
            self.assertEqual(status["total"], 1000)
            self.assertEqual(status["unprocessed"], 1000)
            self.assertEqual(status["processing"], 0)
            self.assertEqual(status["done"], 0)
            self.assertEqual(status["failed"], 0)

            # check state
            state = processor.current_state()
            self.assertSequenceEqual(state["done"], [])
            self.assertSequenceEqual(state["failed"], [])
            self.assertSequenceEqual(state["processing"], [])
            self.assertSequenceEqual(state["unprocessed"], list(range(1000)))

            # process the data
            processor.process(n_workers=4)

            # check final status
            status = processor.status()
            self.assertEqual(status["total"], 1000)
            self.assertEqual(status["unprocessed"], 0)
            self.assertEqual(status["processing"], 0)
            self.assertEqual(status["done"], 800)
            self.assertEqual(status["failed"], 200)

            # check final state
            state = processor.current_state()
            # check state
            state = processor.current_state()
            self.assertSequenceEqual(
                state["done"],
                [i for i in range(1000) if i % 5 != 0],
            )
            self.assertSequenceEqual(state["failed"], list(range(0, 1000, 5)))
            self.assertSequenceEqual(state["processing"], [])
            self.assertSequenceEqual(state["unprocessed"], [])


if __name__ == "__main__":
    run_unittest(UnitTests)
