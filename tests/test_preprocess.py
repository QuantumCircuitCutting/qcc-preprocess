import unittest
from unittest.mock import patch

from qiskit import QuantumCircuit

from circuit_preprocess.benchmark import PipelineResult
from circuit_preprocess.preprocess import (
    available_optimization_methods,
    optimize_circuit,
    optimize_circuit_auto_select,
    optimize_circuit_with_report,
)


def _make_result(twoq_after: int, depth_after: int, runtime_sec: float) -> PipelineResult:
    qc_in = QuantumCircuit(2)
    qc_out = QuantumCircuit(2)
    return PipelineResult(
        name="test",
        qc_in=qc_in,
        qc_out=qc_out,
        twoq_before=1,
        twoq_after=twoq_after,
        depth_before=2,
        depth_after=depth_after,
        runtime_sec=runtime_sec,
    )


class TestPreprocessAPI(unittest.TestCase):
    def test_available_optimization_methods(self) -> None:
        self.assertEqual(
            available_optimization_methods(),
            ("transpile_only", "lc", "partitioned_lc", "partitioned_lc_v2"),
        )

    def test_optimize_circuit_with_report_rejects_non_qiskit_input(self) -> None:
        with self.assertRaises(TypeError):
            optimize_circuit_with_report("not-a-circuit")

    @patch("circuit_preprocess.preprocess.transpile_only_pipeline")
    def test_optimize_circuit_with_report_dispatches_transpile_only(self, mock_pipeline) -> None:
        report = _make_result(twoq_after=1, depth_after=1, runtime_sec=0.1)
        mock_pipeline.return_value = report

        qc = QuantumCircuit(2)
        out = optimize_circuit_with_report(
            qc,
            method="transpile_only",
            optimization_level_after=2,
        )

        self.assertIs(out, report)
        mock_pipeline.assert_called_once()
        args, kwargs = mock_pipeline.call_args
        self.assertIsNot(args[0], qc)
        self.assertEqual(kwargs["optimization_level_after"], 2)
        self.assertEqual(kwargs["name"], "transpile_only")

    @patch("circuit_preprocess.preprocess.lc_pipeline")
    def test_optimize_circuit_with_report_dispatches_lc(self, mock_pipeline) -> None:
        report = _make_result(twoq_after=1, depth_after=1, runtime_sec=0.1)
        mock_pipeline.return_value = report

        qc = QuantumCircuit(3)
        out = optimize_circuit_with_report(
            qc,
            method="lc",
            target_basis=("rz", "sx", "x", "cx"),
            force_cx=False,
            exact_check_qubits_threshold=7,
        )

        self.assertIs(out, report)
        mock_pipeline.assert_called_once()
        _, kwargs = mock_pipeline.call_args
        self.assertEqual(kwargs["target_basis"], ("rz", "sx", "x", "cx"))
        self.assertFalse(kwargs["force_cx"])
        self.assertEqual(kwargs["exact_check_qubits_threshold"], 7)
        self.assertEqual(kwargs["name"], "zx_full_reduce")

    @patch("circuit_preprocess.preprocess.partitioned_lc_pipeline")
    def test_optimize_circuit_with_report_dispatches_partitioned_lc(self, mock_pipeline) -> None:
        report = _make_result(twoq_after=1, depth_after=1, runtime_sec=0.1)
        mock_pipeline.return_value = report

        qc = QuantumCircuit(4)
        out = optimize_circuit_with_report(
            qc,
            method="partitioned_lc",
            processors=2,
            sizes=(2, 2),
            iters=5,
            w_remote=3.0,
        )

        self.assertIs(out, report)
        mock_pipeline.assert_called_once()
        _, kwargs = mock_pipeline.call_args
        self.assertEqual(kwargs["processors"], 2)
        self.assertEqual(kwargs["sizes"], (2, 2))
        self.assertEqual(kwargs["iters"], 5)
        self.assertEqual(kwargs["w_remote"], 3.0)
        self.assertEqual(kwargs["name"], "partitioned_lc_v1")

    @patch("circuit_preprocess.preprocess.partitioned_lc_pipeline_v2")
    def test_optimize_circuit_with_report_dispatches_partitioned_lc_v2(self, mock_pipeline) -> None:
        report = _make_result(twoq_after=1, depth_after=1, runtime_sec=0.1)
        mock_pipeline.return_value = report

        qc = QuantumCircuit(4)
        out = optimize_circuit_with_report(
            qc,
            method="partitioned_lc_v2",
            processors=2,
            sizes=(2, 2),
            w_remote=3.5,
            w_local=1.2,
            max_outer_iters=6,
            max_moves_per_pass=77,
            exact_check_qubits_threshold=11,
        )

        self.assertIs(out, report)
        mock_pipeline.assert_called_once()
        _, kwargs = mock_pipeline.call_args
        self.assertEqual(kwargs["processors"], 2)
        self.assertEqual(kwargs["sizes"], (2, 2))
        self.assertEqual(kwargs["w_remote"], 3.5)
        self.assertEqual(kwargs["w_local"], 1.2)
        self.assertEqual(kwargs["max_outer_iters"], 6)
        self.assertEqual(kwargs["max_moves_per_pass"], 77)
        self.assertEqual(kwargs["exact_check_qubits_threshold"], 11)
        self.assertEqual(kwargs["name"], "partitioned_lc_v2")

    def test_optimize_circuit_with_report_rejects_unknown_method(self) -> None:
        qc = QuantumCircuit(2)
        with self.assertRaises(ValueError) as ctx:
            optimize_circuit_with_report(qc, method="unknown")
        self.assertIn("未知の method", str(ctx.exception))

    @patch("circuit_preprocess.preprocess.optimize_circuit_with_report")
    def test_optimize_circuit_returns_qc_out_only(self, mock_with_report) -> None:
        report = _make_result(twoq_after=1, depth_after=1, runtime_sec=0.1)
        expected_qc = QuantumCircuit(2)
        report.qc_out = expected_qc
        mock_with_report.return_value = report

        qc = QuantumCircuit(2)
        out = optimize_circuit(qc, method="lc", processors=4)

        self.assertIs(out, expected_qc)
        mock_with_report.assert_called_once_with(qc, method="lc", processors=4)

    def test_optimize_circuit_auto_select_rejects_invalid_inputs(self) -> None:
        with self.assertRaises(TypeError):
            optimize_circuit_auto_select("not-a-circuit")

        qc = QuantumCircuit(2)
        with self.assertRaises(ValueError):
            optimize_circuit_auto_select(qc, methods=[])

    @patch("circuit_preprocess.preprocess.optimize_circuit_with_report")
    def test_optimize_circuit_auto_select_chooses_by_twoq_depth_runtime(self, mock_with_report) -> None:
        report_a = _make_result(twoq_after=4, depth_after=10, runtime_sec=1.0)
        report_b = _make_result(twoq_after=3, depth_after=9, runtime_sec=2.0)
        report_c = _make_result(twoq_after=3, depth_after=8, runtime_sec=3.0)
        report_d = _make_result(twoq_after=3, depth_after=8, runtime_sec=2.5)

        def _side_effect(*args, **kwargs):
            method = kwargs["method"]
            mapping = {
                "transpile_only": report_a,
                "lc": report_b,
                "partitioned_lc": report_c,
                "partitioned_lc_v2": report_d,
            }
            return mapping[method]

        mock_with_report.side_effect = _side_effect

        qc = QuantumCircuit(3)
        qc_best, best_method, reports = optimize_circuit_auto_select(qc)

        self.assertEqual(best_method, "partitioned_lc_v2")
        self.assertIs(qc_best, report_d.qc_out)
        self.assertEqual([m for m, _ in reports], list(available_optimization_methods()))
        self.assertEqual(mock_with_report.call_count, 4)


if __name__ == "__main__":
    unittest.main()