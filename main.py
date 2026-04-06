"""Basketball Court Tracker — Entry Point.

Usage:
    python main.py VIDEO_PATH --roboflow-key YOUR_KEY [options]

Example:
    python main.py game_clip.mp4 --roboflow-key abc123 --debug --frame-skip 2
"""

from config import Config
from pipeline import Pipeline


def main():
    config = Config.from_args()
    pipeline = Pipeline(config)
    pipeline.run()


if __name__ == "__main__":
    main()
