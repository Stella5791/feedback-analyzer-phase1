from pathlib import Path
import pandas as pd


# Rutas
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
INPUT_FILE = DATA_DIR / "input_encuestas.csv"


def load_data():
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {INPUT_FILE}")

    df = pd.read_csv(INPUT_FILE)
    return df


def main():
    df = load_data()

    print("\nDataset cargado")
    print("Shape:", df.shape)

    print("\nColumnas:")
    for col in df.columns:
        print("-", col)

    print("\nPreview:")
    print(df.head())


if __name__ == "__main__":
    main()
