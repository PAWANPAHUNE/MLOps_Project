import gradio as gr
import pandas as pd

from src.MLOps_Project.pipeline.prediction_pipeline import PredictionPipeline


history_data = []



def predict_quality(
    fixed_acidity,
    volatile_acidity,
    citric_acid,
    residual_sugar,
    chlorides,
    free_sulfur_dioxide,
    total_sulfur_dioxide,
    density,
    pH,
    sulphates,
    alcohol
):
    try:
        input_dict = {
            "fixed acidity": fixed_acidity,
            "volatile acidity": volatile_acidity,
            "citric acid": citric_acid,
            "residual sugar": residual_sugar,
            "chlorides": chlorides,
            "free sulfur dioxide": free_sulfur_dioxide,
            "total sulfur dioxide": total_sulfur_dioxide,
            "density": density,
            "pH": pH,
            "sulphates": sulphates,
            "alcohol": alcohol,
        }

        df = pd.DataFrame([input_dict])
        pipeline = PredictionPipeline()
        prediction = pipeline.predict(df)[0]

        # Save to history
        history_data.append({**input_dict, "prediction": prediction})

        return f"🍷 Predicted Wine Quality: {prediction}", pd.DataFrame(history_data)

    except Exception as e:
        return f"❌ Error: {str(e)}", pd.DataFrame(history_data)



custom_css = """
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&display=swap');

* {
    font-family: 'JetBrains Mono', monospace;
}

body, .gradio-container {
    background-color: #0b0b0b !important;
}

h1, h2, h3, label {
    color: #facc15 !important;
}

input, textarea {
    background-color: #151515 !important;
    color: white !important;
    border: 1px solid #facc15 !important;
}

button {
    background-color: #facc15 !important;
    color: black !important;
    font-weight: 600;
    border-radius: 6px;
}

button:hover {
    background-color: #fde047 !important;
}

.dataframe {
    background-color: #0b0b0b !important;
    color: white !important;
}
"""



with gr.Blocks(css=custom_css, theme=gr.themes.Base()) as app:

    gr.Markdown(
        """
        # 🍷 Wine Quality Prediction  
        ### End-to-End MLOps Deployment with History Tracking
        """
    )

    with gr.Tabs():

       
        with gr.Tab("🔮 Predict"):
            with gr.Row():
                with gr.Column():
                    fixed_acidity = gr.Number(label="Fixed Acidity")
                    volatile_acidity = gr.Number(label="Volatile Acidity")
                    citric_acid = gr.Number(label="Citric Acid")
                    residual_sugar = gr.Number(label="Residual Sugar")
                    chlorides = gr.Number(label="Chlorides")
                    free_sulfur_dioxide = gr.Number(label="Free Sulfur Dioxide")

                with gr.Column():
                    total_sulfur_dioxide = gr.Number(label="Total Sulfur Dioxide")
                    density = gr.Number(label="Density")
                    pH = gr.Number(label="pH")
                    sulphates = gr.Number(label="Sulphates")
                    alcohol = gr.Number(label="Alcohol")

            predict_btn = gr.Button("🚀 Predict")

            prediction_output = gr.Textbox(
                label="Prediction Result",
                interactive=False
            )

      
        with gr.Tab("📜 History"):
            history_table = gr.Dataframe(
                headers=[
                    "fixed acidity", "volatile acidity", "citric acid",
                    "residual sugar", "chlorides", "free sulfur dioxide",
                    "total sulfur dioxide", "density", "pH",
                    "sulphates", "alcohol", "prediction"
                ],
                interactive=False,
                wrap=True
            )

        predict_btn.click(
            fn=predict_quality,
            inputs=[
                fixed_acidity,
                volatile_acidity,
                citric_acid,
                residual_sugar,
                chlorides,
                free_sulfur_dioxide,
                total_sulfur_dioxide,
                density,
                pH,
                sulphates,
                alcohol,
            ],
            outputs=[prediction_output, history_table]
        )


if __name__ == "__main__":
    app.launch(server_name="0.0.0.0", server_port=7860)
