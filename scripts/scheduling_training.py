#!/usr/bin/env python
import schedule
import time
import logging
from datetime import datetime
import sys
import os

# Add the project root to python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml_pipeline.train import train_all_models

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)


def scheduled_training(retrain_all=False):
    """Scheduled training job"""
    logger.info(f"Starting scheduled training at {datetime.now()}")
    try:
        trained_count = train_all_models(retrain_all=retrain_all)
        logger.info(f"Training completed. Models trained: {trained_count}")
        return trained_count
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return 0



if __name__ == "__main__":
    # Parse command line arguments
    import argparse

    parser = argparse.ArgumentParser(description="Model Training Scheduler")
    parser.add_argument(
        "--retrain-all",
        action="store_true",
        help="Force retrain all models regardless of last training date",
    )
    parser.add_argument(
        "--run-once",
        action="store_true",
        help="Run training once and exit",
    )

    args = parser.parse_args()

    if args.run_once:
        # Run once and exit
        logger.info("Running training once...")
        trained_count = scheduled_training(retrain_all=args.retrain_all)
        logger.info(
            f"Training complete. Exiting. Models trained: {trained_count}"
        )
    else:
        # Schedule training every 2 weeks on Monday at 2 AM
        schedule.every(2).weeks.monday.at("02:00").do(scheduled_training)

        logger.info("Training scheduler started.")
        logger.info("Will run every 2 weeks on Monday at 2 AM.")
        logger.info("Press Ctrl+C to stop.")

        # Also run once now (optional - you can remove this if you don't want it)
        logger.info("Running initial training...")
        scheduled_training(retrain_all=args.retrain_all)

        try:
            while True:
                schedule.run_pending()
                time.sleep(60)  # Check every minute
        except KeyboardInterrupt:
            logger.info("Training scheduler stopped.")
