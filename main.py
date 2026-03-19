from pathlib import Path
import pandas as pd


# Rutas
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
OUTPUT_DIR = BASE_DIR / "output"
INPUT_FILE = DATA_DIR / "input_encuestas.csv"


def load_data() -> pd.DataFrame:
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"No se encontró el archivo: {INPUT_FILE}")

    return pd.read_csv(INPUT_FILE)


def build_comentario_cliente(df: pd.DataFrame) -> pd.DataFrame:
    comment_cols = [
        "Comentario Satisfaccion",
        "Comentario Insatisfaccion",
        "Comentario Mejorar",
    ]

    for col in comment_cols:
        df[col] = df[col].fillna("").astype(str).str.strip()

    df["comentario_cliente"] = df[comment_cols].apply(
        lambda row: " | ".join([text for text in row if text]),
        axis=1
    )

    df["comentario_cliente"] = df["comentario_cliente"].str.strip()

    return df


def derive_tipo_feedback(df: pd.DataFrame) -> pd.DataFrame:
    def classify_row(row):
        if row["Comentario Insatisfaccion"]:
            return "insatisfaccion"
        elif row["Comentario Mejorar"]:
            return "sugerencia"
        elif row["Comentario Satisfaccion"]:
            return "satisfaccion"
        else:
            return "otro"

    df["tipo_feedback"] = df.apply(classify_row, axis=1)
    return df


def sample_comments(df: pd.DataFrame, n: int = 200) -> pd.DataFrame:
    df_valid = df[
        (df["comentario_cliente"] != "") &
        (df["comentario_cliente"].str.len() > 15)
    ].copy()
    return df_valid.sample(n=min(n, len(df_valid)), random_state=42)


def detect_entidad(comment: str) -> str:
    comment_lower = comment.lower()

    if any(word in comment_lower for word in ["producto", "productos", "variedad", "marca", "marcas", "no encuentro", "sin stock", "faltaba", "chocolate", "chocolates"]):
        return "producto"
    if any(word in comment_lower for word in ["caja", "cajero", "cajera"]):
        return "caja"
    if any(word in comment_lower for word in ["precio", "precios", "oferta", "ofertas", "tarjeta"]):
        return "precio"
    if any(word in comment_lower for word in ["atencion", "atención", "personal", "empleado", "empleada", "trato"]):
        return "personal"
    if any(word in comment_lower for word in ["local", "limpio", "ordenado", "limpieza", "aire acondicionado", "wi fi"]):
        return "tienda"

    return "general"


def detect_dimension(comment: str) -> str:
    comment_lower = comment.lower()

    if any(word in comment_lower for word in ["derretido", "derretidos", "calor"]):
        return "cadena_frio"
    if any(word in comment_lower for word in ["no encuentro", "faltaba", "sin stock"]):
        return "stock"
    if any(word in comment_lower for word in ["atencion", "atención", "trato", "resolvió", "resolvio", "amable", "amables"]):
        return "atencion"
    if any(word in comment_lower for word in ["caja", "espera", "fila", "cajero", "cajera"]):
        return "espera_caja"
    if any(word in comment_lower for word in ["precio", "precios"]):
        return "precio"
    if any(word in comment_lower for word in ["limpio", "sucio", "limpieza"]):
        return "limpieza"

    return "general"


def analyze_comment_mock(comment: str) -> dict:
    """
    Simula la respuesta del analizador GPT con la estructura JSON final esperada.
    """
    comment_lower = comment.lower()

    sentiment = "neutral"
    sentiment_intensity = "baja"
    target_entidad_1 = detect_entidad(comment)
    target_dimension_1 = detect_dimension(comment)
    appraisal_tipo_1 = "apreciacion"
    appraisal_valor_1 = "neutral"
    impacto_operativo = "bajo"
    riesgo_alto = "no"
    riesgo_motivo = ""
    cohesion_tipo = "simple"

    positive_words = ["excelente", "bueno",
                      "buenos", "amable", "amables", "sensacional"]
    negative_words = ["malo", "mala", "demora", "espera", "caro", "faltaba",
                      "sucio", "problema", "derretidos", "derretido", "no encuentro"]

    pos_count = sum(1 for word in positive_words if word in comment_lower)
    neg_count = sum(1 for word in negative_words if word in comment_lower)

    if pos_count > neg_count:
        sentiment = "positivo"
        sentiment_intensity = "media"
        appraisal_valor_1 = "positivo"
        appraisal_tipo_1 = "afecto"
        impacto_operativo = "medio"

    elif neg_count > pos_count:
        sentiment = "negativo"
        sentiment_intensity = "media"
        appraisal_valor_1 = "negativo"
        appraisal_tipo_1 = "apreciacion"
        impacto_operativo = "medio"

    elif pos_count > 0 and neg_count > 0:
        sentiment = "mixto"
        appraisal_valor_1 = "mixto"

    if len(comment.split()) > 8:
        cohesion_tipo = "compuesto"

    if sentiment == "negativo" and any(word in comment_lower for word in ["problema", "demora", "espera", "no encuentro"]):
        riesgo_alto = "si"
        riesgo_motivo = "posible friccion operativa"

    return {
        "sentiment": sentiment,
        "sentiment_intensity": sentiment_intensity,
        "target_entidad_1": target_entidad_1,
        "target_dimension_1": target_dimension_1,
        "appraisal_tipo_1": appraisal_tipo_1,
        "appraisal_valor_1": appraisal_valor_1,
        "target_entidad_2": "",
        "target_dimension_2": "",
        "appraisal_tipo_2": "",
        "appraisal_valor_2": "",
        "cohesion_tipo": cohesion_tipo,
        "impacto_operativo": impacto_operativo,
        "riesgo_alto": riesgo_alto,
        "riesgo_motivo": riesgo_motivo,
        "comentario_resumen": comment[:80].strip()
    }


def enrich_with_analysis(df: pd.DataFrame) -> pd.DataFrame:
    df = df.reset_index(drop=True)

    results = df["comentario_cliente"].apply(analyze_comment_mock)
    results_df = pd.json_normalize(results).reset_index(drop=True)

    return pd.concat([df, results_df], axis=1)


def export_csv(df: pd.DataFrame, filename: str = "retail_analyzer_output.csv") -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)

    columns_to_export = [
        "comentario_cliente",
        "tipo_feedback",
        "sentiment",
        "sentiment_intensity",
        "target_entidad_1",
        "target_dimension_1",
        "appraisal_tipo_1",
        "appraisal_valor_1",
        "target_entidad_2",
        "target_dimension_2",
        "appraisal_tipo_2",
        "appraisal_valor_2",
        "cohesion_tipo",
        "impacto_operativo",
        "riesgo_alto",
        "riesgo_motivo",
        "comentario_resumen",
    ]

    df_export = df[columns_to_export].copy()

    output_path = OUTPUT_DIR / filename
    df_export.to_csv(output_path, index=False, encoding="utf-8-sig")

    return output_path


def main():
    df = load_data()
    df = build_comentario_cliente(df)
    df = derive_tipo_feedback(df)

    df_sample = sample_comments(df, n=200)
    df_enriched = enrich_with_analysis(df_sample)

    output_path = export_csv(df_enriched)

    print("\nDataset enriquecido:")
    print("Shape:", df_enriched.shape)

    print("\nPreview enriquecido:")
    preview_cols = [
        "comentario_cliente",
        "tipo_feedback",
        "sentiment",
        "target_entidad_1",
        "target_dimension_1",
        "comentario_resumen"
    ]

    print(df_enriched[preview_cols].head(10).to_string(index=False))

    print(f"\nCSV exportado en: {output_path}")


if __name__ == "__main__":
    main()
