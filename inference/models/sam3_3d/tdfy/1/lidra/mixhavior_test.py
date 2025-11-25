import unittest
import pickle as pkl

from lidra.test.util import run_unittest, temporary_file
import lidra.mixhavior as mixhavior


class A:
    def __init__(self):
        self.x = 0


class BehaviorX(mixhavior.Behavior):
    def __init__(self, x):
        super().__init__()
        self._x = x

    def run(self, x=None):
        x = self._x if x is None else x
        return self.obj.x + x


class BehaviorY(mixhavior.Behavior):
    def string(self):
        return f"x is {self.obj.x}"


class BehaviorZ(BehaviorX, BehaviorY):
    pass


class BehaviorState(mixhavior.Behavior):
    def _setup(self, mixh):
        self.x = mixh.obj.x + 1

    def _cleanup(self):
        delattr(self, "x")

    def _attach(self, mixh):
        self.state = "attached"

    def _detach(self):
        self.state = "detached"

    def __call__(self):
        x = self.mixh.obj.x
        return f"{self.state} state with x = {self.x} (vs obj.x = {x})"


class UnitTests(unittest.TestCase):
    def test_attach_detach(self):
        a = A()
        behavior = BehaviorState()

        self.assertEqual(behavior.is_attached, False)
        self.assertEqual(behavior.is_setup, False)
        with self.assertRaises(mixhavior.DetachedState):
            behavior()

        mixh = mixhavior.get_mixhavior(a)
        bid = mixh.equip(behavior, "state")

        self.assertEqual(bid in mixh.behaviors, True)
        self.assertEqual(bid, "state")
        self.assertEqual(behavior.is_attached, True)
        self.assertEqual(behavior.is_setup, True)
        self.assertEqual(behavior.state, "attached")
        self.assertEqual(behavior.x, 1)
        self.assertEqual(behavior(), "attached state with x = 1 (vs obj.x = 0)")

        popped_behavior = mixh.unequip(bid)
        self.assertEqual(bid in mixh.behaviors, False)
        self.assertEqual(popped_behavior is behavior, True)
        self.assertEqual(behavior.is_attached, False)
        self.assertEqual(behavior.is_setup, True)
        self.assertEqual(behavior.state, "detached")
        self.assertEqual(behavior.x, 1)
        with self.assertRaises(mixhavior.DetachedState):
            behavior()

    def test_mixhavior_basic(self):
        # TODO(Pierre) write better unit tests (bing in pytest fixtures ?)
        a = A()
        bx1 = BehaviorX(1)
        bx42 = BehaviorX(42)
        by = BehaviorY()
        bz = BehaviorZ(2)

        mixh = mixhavior.get_mixhavior(a)
        mixh.equip(bx1, "bx_1")
        mixh.equip(bx42, "bx_2")
        mixh.equip(by, "by")
        mixh.equip(bz, "bz")

        with self.assertRaises(RuntimeError):
            mixh.equip(bx1, "bx_1")

        # check simple calls
        self.assertEqual(mixhavior.get_behavior(a, "by").string(), "x is 0")
        self.assertEqual(mixhavior.get_behavior(a, "bx_1").run(), 1)
        self.assertEqual(mixhavior.get_behavior(a, "bx_2").run(), 42)
        self.assertEqual(mixhavior.get_behavior(a, "bz").string(), "x is 0")
        self.assertEqual(mixhavior.get_behavior(a, "bz").run(), 2)

        a.x = 1

        # all should be updated (i.e. no copy of a)
        self.assertEqual(mixhavior.get_behavior(a, "by").string(), "x is 1")
        self.assertEqual(mixhavior.get_behavior(a, "bx_1").run(), 2)
        self.assertEqual(mixhavior.get_behavior(a, "bx_2").run(), 43)
        self.assertEqual(mixhavior.get_behavior(a, "bz").string(), "x is 1")
        self.assertEqual(mixhavior.get_behavior(a, "bz").run(), 3)

        # check serialization
        with temporary_file() as tmp_path:
            with open(tmp_path, "wb") as f:
                pkl.dump(a, f)
            with open(tmp_path, "rb") as f:
                b = pkl.load(f)

        # loaded object should have same behaviors
        self.assertEqual(mixhavior.get_behavior(b, "by").string(), "x is 1")
        self.assertEqual(mixhavior.get_behavior(b, "bx_1").run(), 2)
        self.assertEqual(mixhavior.get_behavior(b, "bx_2").run(), 43)
        self.assertEqual(mixhavior.get_behavior(b, "bz").string(), "x is 1")
        self.assertEqual(mixhavior.get_behavior(b, "bz").run(), 3)

        # check mixcalling and find_xxx functions
        self.assertEqual(None, mixhavior.Mixcaller(None).do_not_exist_fn())
        self.assertEqual({}, mixhavior.Mixcaller({}).do_not_exist_fn())

        behaviors = mixh.find_all_of_instance(BehaviorY)
        results = mixhavior.Mixcaller(behaviors).string()
        self.assertEqual(results, {"by": "x is 1", "bz": "x is 1"})

        behaviors = mixh.find_all_of_instance(BehaviorX)
        results = mixhavior.Mixcaller(behaviors).run()
        self.assertEqual(results, {"bx_1": 2, "bx_2": 43, "bz": 3})

        # check tuple
        tuple_behaviors = tuple(behaviors.values())
        results = mixhavior.Mixcaller(tuple_behaviors).run()
        self.assertEqual(results, (2, 43, 3))

        behaviors = mixh.find_all_of_prefix("bx_")
        results = mixhavior.Mixcaller(behaviors).run()
        self.assertEqual(results, {"bx_1": 2, "bx_2": 43})

        behaviors = mixh.find_all_of_instance(BehaviorX)
        results = mixhavior.Mixcaller(behaviors).run(2)
        self.assertEqual(results, {"bx_1": 3, "bx_2": 3, "bz": 3})

        bx1.remove()

        behaviors = mixh.find_all_of_instance(BehaviorX)
        results = mixhavior.Mixcaller(behaviors).run()
        self.assertEqual(results, {"bx_2": 43, "bz": 3})

        mixh.unequip("bz")

        behaviors = mixh.find_all_of_instance(BehaviorX)
        results = mixhavior.Mixcaller(behaviors).run()
        self.assertEqual(results, {"bx_2": 43})


if __name__ == "__main__":
    run_unittest(UnitTests)
