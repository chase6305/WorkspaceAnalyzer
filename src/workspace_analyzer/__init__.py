"""Robot workspace analysis without simulator coupling."""

__version__ = "0.1.0"

from .analyzer import (
    AnalysisCancelled,
    AnalysisResult,
    CartesianConfig,
    WorkspaceAnalyzer,
    WorkspaceConfig,
)
from .boundary import BoundaryConfig, BoundaryResult, refine_translation_boundary
from .cache import ResultCache, analysis_cache_key
from .coverage import OrientationCoverage, summarize_orientation_coverage
from .diversity import IKDiversityConfig, make_ik_seed_trials, summarize_ik_diversity
from .kinematics import IKResult, KinematicsSolver, SolverConfig, create_solver
from .metrics import DexterityResult, dexterity_metrics
from .model import Joint, JointLimit, RobotModel
from .reachability import ReachabilityConfig, reassess_quality
from .robustness import PosePerturbations
from .sampling import SamplingConfig, SamplingStrategy
from .selection import select_ik_solutions
from .stability import IKStabilityResult, IKTrial, analyze_ik_stability
from .trajectory import TrajectoryIKResult, solve_trajectory

__all__ = [
    "__version__",
    "AnalysisResult",
    "AnalysisCancelled",
    "BoundaryConfig",
    "BoundaryResult",
    "CartesianConfig",
    "DexterityResult",
    "IKResult",
    "IKDiversityConfig",
    "IKTrial",
    "IKStabilityResult",
    "OrientationCoverage",
    "PosePerturbations",
    "Joint",
    "JointLimit",
    "KinematicsSolver",
    "RobotModel",
    "ReachabilityConfig",
    "ResultCache",
    "SamplingConfig",
    "SamplingStrategy",
    "SolverConfig",
    "WorkspaceAnalyzer",
    "WorkspaceConfig",
    "TrajectoryIKResult",
    "create_solver",
    "analysis_cache_key",
    "dexterity_metrics",
    "solve_trajectory",
    "reassess_quality",
    "summarize_orientation_coverage",
    "refine_translation_boundary",
    "analyze_ik_stability",
    "select_ik_solutions",
    "make_ik_seed_trials",
    "summarize_ik_diversity",
]
