"""python scripts/run.py --config configs/smoke.yaml [--models stub deepseek-chat] [--stages llm,exec,score]"""
from gsi.experiment.run import main

if __name__ == "__main__":
    main()
