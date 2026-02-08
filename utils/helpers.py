from datetime import datetime
import pytz


def get_current_datetime():
    """Get current datetime in UTC"""
    timezone = pytz.timezone("UTC")
    return datetime.now(timezone)


def format_accuracy_class(mape):
    """Get CSS class for accuracy"""
    if mape <= 10:
        return "accuracy-good"
    elif mape <= 20:
        return "accuracy-fair"
    else:
        return "accuracy-poor"


def format_accuracy_text(mape):
    """Get text for accuracy"""
    if mape <= 10:
        return "Excellent"
    elif mape <= 20:
        return "Good"
    else:
        return "Fair"
