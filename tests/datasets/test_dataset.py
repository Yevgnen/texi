# -*- coding: utf-8 -*-

import unittest

from texi.datasets.dataset import DatasetTransformMixin


class TestDatasetTransformMixin(unittest.TestCase):
    def setUp(self):
        class DummyMixin(DatasetTransformMixin):
            _mixin_attributes = ["x"]
            _mixin_transform = "transform"
            _mixin_inverse_transform = "inverse_transform"

            def transform(self):
                self._check_transform()

                self.x = 1

            def inverse_transform(self):
                self._check_inverse_transform()
                self._remove_attributes()

            def __iter__(self):
                pass

            def __getitem__(self):
                pass

        self.mixin = DummyMixin()

    def test_transform(self):
        self.assertIsNone(getattr(self.mixin, "x", None))
        self.mixin.transform()
        self.assertIsNotNone(getattr(self.mixin, "x"))
        with self.assertRaises(RuntimeError) as ctx:
            self.mixin.transform()
        self.assertEqual(str(ctx.exception), "Can not call `.transform()` twice.")
        self.mixin.inverse_transform()
        self.assertIsNone(getattr(self.mixin, "x", None))

    def test_inverse_transform(self):
        self.mixin.transform()
        self.assertIsNotNone(getattr(self.mixin, "x"))
        self.mixin.inverse_transform()
        self.assertIsNone(getattr(self.mixin, "x", None))
        with self.assertRaises(RuntimeError) as ctx:
            self.mixin.inverse_transform()
        self.assertEqual(
            str(ctx.exception),
            "Can not call `.inverse_transform()` before `.transform()`.",
        )


if __name__ == "__main__":
    unittest.main()
