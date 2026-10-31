"""Compile the notebook callbacks with Numba; no VectorBT or market data needed."""
import ast
from collections import namedtuple
import json
from pathlib import Path
import unittest

from numba import njit
from numba.core.registry import CPUDispatcher
import numpy as np


NOTEBOOK = json.loads(
    (Path(__file__).resolve().parents[1] / "VectorBT Backtest.ipynb").read_text()
)
SLContext = namedtuple("SLContext", "position_now init_price val_price_now curr_stop curr_trail")
TPContext = namedtuple("TPContext", "position_now init_price val_price_now curr_stop")


def functions():
    namespace = {"np": np, "njit": njit}
    wanted = {"adjust_sl_func_nb", "adjust_tp_func_nb", "apply_margin"}
    for cell in NOTEBOOK["cells"]:
        source = "".join(cell.get("source", []))
        if cell["cell_type"] != "code" or not any(f"def {name}(" in source for name in wanted):
            continue
        nodes = [node for node in ast.parse(source).body
                 if isinstance(node, ast.FunctionDef) and node.name in wanted]
        exec(compile(ast.Module(body=nodes, type_ignores=[]), "<notebook>", "exec"), namespace)
    return namespace


CALLBACKS = functions()


class StopCallbackTests(unittest.TestCase):
    def test_stop_loss_thresholds_work_for_long_and_short_gains(self):
        thresholds = [(0.19, 0.03, False), (0.20, 0.07, True),
                      (0.24, 0.07, True), (0.25, 0.15, True),
                      (0.39, 0.15, True), (0.40, 0.25, True),
                      (0.60, 0.25, True)]
        for position in (1.0, -1.0):
            for profit, stop, trail in thresholds:
                price = 100.0 + (profit * 100 if position > 0 else -profit * 100)
                with self.subTest(position=position, profit=profit):
                    result = CALLBACKS["adjust_sl_func_nb"](SLContext(position, 100.0, price, 0.03, False))
                    self.assertEqual(result, (stop, trail))

    def test_take_profit_threshold_works_for_long_and_short_gains(self):
        for position in (1.0, -1.0):
            for profit, expected in ((0.49, 0.10), (0.50, 0.55), (0.60, 0.55)):
                price = 100.0 + (profit * 100 if position > 0 else -profit * 100)
                with self.subTest(position=position, profit=profit):
                    self.assertEqual(CALLBACKS["adjust_tp_func_nb"](
                        TPContext(position, 100.0, price, 0.10)), expected)

    def test_losses_preserve_both_current_stops_and_trailing_state(self):
        for position, price in ((1.0, 50.0), (-1.0, 150.0)):
            for trail in (False, True):
                with self.subTest(position=position, trail=trail):
                    self.assertEqual(CALLBACKS["adjust_sl_func_nb"](
                        SLContext(position, 100.0, price, 0.03, trail)), (0.03, trail))
                    self.assertEqual(CALLBACKS["adjust_tp_func_nb"](
                        TPContext(position, 100.0, price, 0.10)), 0.10)

    def test_flat_missing_and_invalid_contexts_preserve_existing_settings(self):
        contexts = [(0.0, 100.0, 160.0), (np.nan, 100.0, 160.0),
                    (np.inf, 100.0, 160.0), (1.0, 0.0, 160.0),
                    (-1.0, -100.0, 50.0), (1.0, np.nan, 160.0),
                    (1.0, np.inf, 160.0), (1.0, 100.0, np.nan),
                    (-1.0, 100.0, np.inf),
                    (1.0, np.finfo(float).tiny, np.finfo(float).max)]
        for position, initial, price in contexts:
            with self.subTest(position=position, initial=initial, price=price):
                self.assertEqual(CALLBACKS["adjust_sl_func_nb"](
                    SLContext(position, initial, price, 0.03, True)), (0.03, True))
                self.assertEqual(CALLBACKS["adjust_tp_func_nb"](
                    TPContext(position, initial, price, 0.10)), 0.10)

    def test_position_size_does_not_change_percentage_thresholds(self):
        for position, price in ((1.0, 150.0), (100.0, 150.0), (-1.0, 50.0), (-100.0, 50.0)):
            self.assertEqual(CALLBACKS["adjust_sl_func_nb"](
                SLContext(position, 100.0, price, 0.03, False)), (0.25, True))
            self.assertEqual(CALLBACKS["adjust_tp_func_nb"](
                TPContext(position, 100.0, price, 0.10)), 0.55)

    def test_disabled_stop_is_preserved_for_a_flat_position(self):
        stop, trail = CALLBACKS["adjust_sl_func_nb"](
            SLContext(0.0, 100.0, 160.0, np.nan, False))
        self.assertTrue(np.isnan(stop))
        self.assertFalse(trail)
        self.assertTrue(np.isnan(CALLBACKS["adjust_tp_func_nb"](
            TPContext(0.0, 100.0, 160.0, np.nan))))

    def test_callbacks_are_actually_compiled_in_numba_nopython_mode(self):
        CALLBACKS["adjust_sl_func_nb"](SLContext(1.0, 100.0, 120.0, 0.03, False))
        CALLBACKS["adjust_tp_func_nb"](TPContext(-1.0, 100.0, 50.0, 0.10))
        for name in ("adjust_sl_func_nb", "adjust_tp_func_nb"):
            self.assertIsInstance(CALLBACKS[name], CPUDispatcher)
            self.assertTrue(CALLBACKS[name].nopython_signatures)

    def test_benchmark_repetitions_receive_fresh_entry_arrays(self):
        source = next("".join(cell["source"]) for cell in NOTEBOOK["cells"]
                      if "%timeit apply_margin(" in "".join(cell.get("source", [])))
        timed = next(line.removeprefix("%timeit ") for line in source.splitlines()
                     if line.startswith("%timeit "))
        setup = "\n".join(line for line in source.splitlines() if not line.startswith("%timeit "))
        original = np.array([True, False, False, False])
        received, results = [], []
        class Series:
            def __init__(self, values):
                self.values = values

            def copy(self):
                return Series(self.values.copy())

        def apply_margin(prices, signals, margin):
            received.append(signals)
            np.testing.assert_array_equal(signals, original)
            CALLBACKS["apply_margin"](prices, signals, margin)
            results.append(signals.copy())
        namespace = {"apply_margin": apply_margin,
                     "price_close": Series(np.array([100.0, 102.0, 104.0, 106.0])),
                     "long_entries": Series(original)}
        exec(setup, namespace)
        exec(timed, namespace)
        exec(timed, namespace)
        self.assertIsNot(received[0], received[1])
        np.testing.assert_array_equal(original, [True, False, False, False])
        np.testing.assert_array_equal(results[0], results[1])


if __name__ == "__main__":
    unittest.main()
