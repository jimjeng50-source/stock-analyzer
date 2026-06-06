try:
    from .claude_api import get_investment_advice
except ImportError:
    def get_investment_advice(*args, **kwargs):
        return "AI 建議模組未安裝"

try:
    from .report import save_html_report
except ImportError:
    def save_html_report(*args, **kwargs):
        pass

from .risk_correlation import compute_risk_correlations

__all__ = ["get_investment_advice", "save_html_report", "compute_risk_correlations"]
