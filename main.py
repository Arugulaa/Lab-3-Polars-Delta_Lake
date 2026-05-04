# main.py
import sys
import traceback

# Импортируем функции запуска из каждого слоя
from src.bronze import run_bronze
from src.silver import run_silver
from src.gold   import run_gold
from src.ml     import run_ml
from src.delta_ops import run_delta_ops
import logging
import os


# Настраиваем логгер: пишет и в консоль, и в файл logs/pipeline.log
os.makedirs("logs", exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("logs/pipeline.log"),  # запись в файл
        logging.StreamHandler()                     # вывод в консоль
    ]
)
logger = logging.getLogger(__name__)


def main():
    # Список шагов: (название для вывода, функция)
    steps = [
        ("BRONZE", run_bronze),
        ("SILVER", run_silver),
        ("GOLD",   run_gold),
        ("DELTA OPS", run_delta_ops),
        ("ML",     run_ml),
    ]
    logger.info("Пайплайн запущен")
    for step_name, step_func in steps:
        logger.info(f"{'='*40}")
        logger.info(f"Запуск шага: {step_name}")
        logger.info(f"{'='*40}")

        try:
            step_func()  # вызываем функцию шага
            logger.info(f"{step_name}: OK")
        except Exception as e:
            # Если шаг упал — печатаем ошибку и останавливаем весь пайплайн
            logger.error(f"{step_name}: ОШИБКА — {e}")
            logger.error(traceback.format_exc())
            sys.exit(1)  # выходим с кодом ошибки (Docker увидит что контейнер упал)

    logger.info("Весь пайплайн завершён успешно!")

if __name__ == "__main__":
    main()