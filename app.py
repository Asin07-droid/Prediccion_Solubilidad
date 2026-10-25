
"""Predict Solubility — interfaz científica.
La interfaz carga el modelo guardado y no lo reentrena.
"""

from pathlib import Path
import math
import tkinter as tk
from tkinter import ttk, messagebox

import joblib
import pandas as pd
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


# ============================================================
# CONFIGURACIÓN
# ============================================================

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "resultados" / "modelos_finales"

MODEL_PATH = MODEL_DIR / "modelo_potencia_bruta.pkl"
FEATURES_PATH = MODEL_DIR / "features_potencia_bruta.pkl"

COLORS = {
    "background": "#F4F7FB",
    "surface": "#FFFFFF",
    "primary": "#1769AA",
    "primary_dark": "#104C7D",
    "primary_light": "#EAF3FB",
    "text": "#243447",
    "muted": "#6B7C93",
    "border": "#DFE7F0",
    "success": "#24865B",
    "error": "#B93838",
}


# ============================================================
# MODELO Y PREDICCIÓN
# ============================================================

def load_artifacts():
    """Carga el modelo existente y sus características."""

    for path in (MODEL_PATH, FEATURES_PATH):
        if not path.is_file():
            raise FileNotFoundError(
                f"No se encontró el archivo:\n{path}"
            )

    model = joblib.load(MODEL_PATH)
    features = joblib.load(FEATURES_PATH)

    if not isinstance(features, (list, tuple)) or not features:
        raise ValueError(
            "La lista de características no es válida."
        )

    return model, list(features)


def predict_solubility(model, features, smiles):
    """Calcula los descriptores y obtiene la predicción."""

    smiles = smiles.strip()

    if not smiles:
        raise ValueError("Introduce una cadena SMILES.")

    mol = Chem.MolFromSmiles(smiles)

    if mol is None:
        raise ValueError(
            "La estructura SMILES no es válida. "
            "Comprueba la cadena e inténtalo de nuevo."
        )

    descriptors = Descriptors.CalcMolDescriptors(mol)

    values = {}

    for feature in features:
        if feature not in descriptors:
            raise ValueError(
                f"RDKit no calculó el descriptor: {feature}"
            )

        value = descriptors[feature]

        if value is None or not math.isfinite(float(value)):
            raise ValueError(
                f"El descriptor {feature} no tiene "
                "un valor numérico válido."
            )

        values[feature] = float(value)

    X = pd.DataFrame([values], columns=features)

    log_s = float(model.predict(X)[0])

    if not math.isfinite(log_s):
        raise ValueError(
            "El modelo produjo un resultado no finito."
        )

    # logS se interpreta como log10 de la solubilidad en mol/L.
    if -300 <= log_s <= 300:
        solubility = f"{10 ** log_s:.4g} mol/L"
    else:
        solubility = "Fuera del intervalo numérico representable"

    formula = rdMolDescriptors.CalcMolFormula(mol)

    return log_s, solubility, formula


# ============================================================
# INTERFAZ GRÁFICA
# ============================================================

class SolubilityApp:

    def __init__(self, root, model, features):

        self.root = root
        self.model = model
        self.features = features

        self.root.title("Predict Solubility")
        self.root.geometry("760x620")
        self.root.minsize(620, 560)
        self.root.configure(bg=COLORS["background"])

        self.smiles_var = tk.StringVar()
        self.status_var = tk.StringVar(
            value="Modelo listo para realizar predicciones"
        )

        self._configure_styles()
        self._build_interface()

        self.root.bind("<Return>", self._on_enter)

    # --------------------------------------------------------
    # ESTILOS
    # --------------------------------------------------------

    def _configure_styles(self):

        style = ttk.Style()

        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        style.configure(
            "Primary.TButton",
            font=("Segoe UI", 10, "bold"),
            foreground="#FFFFFF",
            background=COLORS["primary"],
            padding=(18, 11),
            borderwidth=0
        )

        style.map(
            "Primary.TButton",
            background=[
                ("pressed", COLORS["primary_dark"]),
                ("active", COLORS["primary_dark"]),
                ("disabled", "#A9C5DC")
            ],
            foreground=[("disabled", "#FFFFFF")]
        )

        style.configure(
            "Secondary.TButton",
            font=("Segoe UI", 10),
            foreground=COLORS["text"],
            background="#E9EFF6",
            padding=(18, 11),
            borderwidth=0
        )

        style.map(
            "Secondary.TButton",
            background=[
                ("pressed", "#D7E2ED"),
                ("active", "#DDE7F1")
            ]
        )

    # --------------------------------------------------------
    # CONSTRUCCIÓN VISUAL
    # --------------------------------------------------------

    def _build_interface(self):

        # Cabecera
        header = tk.Frame(
            self.root,
            bg=COLORS["primary_dark"],
            padx=30,
            pady=22
        )
        header.pack(fill="x")

        brand_row = tk.Frame(
            header,
            bg=COLORS["primary_dark"]
        )
        brand_row.pack(fill="x")

        logo = tk.Label(
            brand_row,
            text="PS",
            font=("Segoe UI", 14, "bold"),
            bg="#FFFFFF",
            fg=COLORS["primary_dark"],
            width=3,
            height=1,
            padx=2,
            pady=5
        )
        logo.pack(side="left", padx=(0, 13))

        brand = tk.Frame(
            brand_row,
            bg=COLORS["primary_dark"]
        )
        brand.pack(side="left", fill="x", expand=True)

        tk.Label(
            brand,
            text="PREDICT SOLUBILITY",
            font=("Segoe UI", 18, "bold"),
            bg=COLORS["primary_dark"],
            fg="#FFFFFF"
        ).pack(anchor="w")

        tk.Label(
            brand,
            text="Molecular property prediction",
            font=("Segoe UI", 10),
            bg=COLORS["primary_dark"],
            fg="#D7E8F7"
        ).pack(anchor="w", pady=(2, 0))

        # Contenido principal
        content = tk.Frame(
            self.root,
            bg=COLORS["background"],
            padx=28,
            pady=24
        )
        content.pack(fill="both", expand=True)

        # Tarjeta de entrada
        input_card = self._create_card(content)
        input_card.pack(fill="x")

        tk.Label(
            input_card,
            text="Entrada molecular",
            font=("Segoe UI", 12, "bold"),
            bg=COLORS["surface"],
            fg=COLORS["text"]
        ).pack(anchor="w")

        tk.Label(
            input_card,
            text=(
                "Introduce la representación SMILES "
                "de la molécula que deseas evaluar."
            ),
            font=("Segoe UI", 9),
            bg=COLORS["surface"],
            fg=COLORS["muted"],
            wraplength=650,
            justify="left"
        ).pack(anchor="w", pady=(5, 14))

        self.smiles_entry = tk.Entry(
            input_card,
            textvariable=self.smiles_var,
            font=("Consolas", 11),
            bg="#FBFDFF",
            fg=COLORS["text"],
            insertbackground=COLORS["primary"],
            relief="flat",
            highlightthickness=1,
            highlightbackground=COLORS["border"],
            highlightcolor=COLORS["primary"]
        )
        self.smiles_entry.pack(fill="x", ipady=10)

        # Botones con distribución uniforme
        button_row = tk.Frame(
            input_card,
            bg=COLORS["surface"]
        )
        button_row.pack(fill="x", pady=(15, 0))

        button_row.columnconfigure(0, weight=3)
        button_row.columnconfigure(1, weight=1)

        self.predict_button = ttk.Button(
            button_row,
            text="Predecir solubilidad",
            style="Primary.TButton",
            command=self.predict
        )
        self.predict_button.grid(
            row=0, column=0, sticky="ew", padx=(0, 8)
        )

        ttk.Button(
            button_row,
            text="Limpiar",
            style="Secondary.TButton",
            command=self.clear
        ).grid(row=0, column=1, sticky="ew")

        # Tarjeta de resultados
        result_card = self._create_card(content)
        result_card.pack(
            fill="both",
            expand=True,
            pady=(18, 0)
        )

        result_heading = tk.Frame(
            result_card,
            bg=COLORS["surface"]
        )
        result_heading.pack(fill="x")

        tk.Label(
            result_heading,
            text="Resultado de la predicción",
            font=("Segoe UI", 12, "bold"),
            bg=COLORS["surface"],
            fg=COLORS["text"]
        ).pack(side="left")

        self.result_badge = tk.Label(
            result_heading,
            text="EN ESPERA",
            font=("Segoe UI", 8, "bold"),
            bg=COLORS["primary_light"],
            fg=COLORS["primary"],
            padx=9,
            pady=5
        )
        self.result_badge.pack(side="right")

        # Panel destacado de logS
        value_panel = tk.Frame(
            result_card,
            bg=COLORS["primary_light"],
            padx=18,
            pady=14
        )
        value_panel.pack(fill="x", pady=(17, 15))

        tk.Label(
            value_panel,
            text="logS",
            font=("Segoe UI", 10, "bold"),
            bg=COLORS["primary_light"],
            fg=COLORS["primary_dark"]
        ).pack(anchor="w")

        self.log_s_label = tk.Label(
            value_panel,
            text="—",
            font=("Segoe UI", 29, "bold"),
            bg=COLORS["primary_light"],
            fg=COLORS["primary_dark"]
        )
        self.log_s_label.pack(anchor="w", pady=(2, 0))

        tk.Label(
            value_panel,
            text="log₁₀ de la solubilidad expresada en mol/L",
            font=("Segoe UI", 9),
            bg=COLORS["primary_light"],
            fg=COLORS["muted"]
        ).pack(anchor="w", pady=(2, 0))

        # Detalles de la predicción
        details = tk.Frame(
            result_card,
            bg=COLORS["surface"]
        )
        details.pack(fill="x")

        tk.Label(
            details,
            text="Solubilidad estimada",
            font=("Segoe UI", 9),
            bg=COLORS["surface"],
            fg=COLORS["muted"]
        ).grid(row=0, column=0, sticky="w", pady=5)

        self.solubility_label = tk.Label(
            details,
            text="—",
            font=("Segoe UI", 11, "bold"),
            bg=COLORS["surface"],
            fg=COLORS["text"]
        )
        self.solubility_label.grid(
            row=0, column=1, sticky="e", pady=5
        )

        tk.Label(
            details,
            text="Fórmula molecular",
            font=("Segoe UI", 9),
            bg=COLORS["surface"],
            fg=COLORS["muted"]
        ).grid(row=1, column=0, sticky="w", pady=5)

        self.formula_label = tk.Label(
            details,
            text="—",
            font=("Segoe UI", 11, "bold"),
            bg=COLORS["surface"],
            fg=COLORS["text"]
        )
        self.formula_label.grid(
            row=1, column=1, sticky="e", pady=5
        )

        details.columnconfigure(0, weight=1)
        details.columnconfigure(1, weight=1)

        # Barra inferior de estado
        footer = tk.Frame(
            self.root,
            bg=COLORS["background"],
            padx=28,
            pady=10
        )
        footer.pack(fill="x")

        self.status_dot = tk.Label(
            footer,
            text="●",
            font=("Segoe UI", 9),
            bg=COLORS["background"],
            fg=COLORS["success"]
        )
        self.status_dot.pack(side="left", padx=(0, 7))

        tk.Label(
            footer,
            textvariable=self.status_var,
            font=("Segoe UI", 9),
            bg=COLORS["background"],
            fg=COLORS["muted"]
        ).pack(side="left", fill="x", expand=True)

        tk.Label(
            footer,
            text=f"{len(self.features)} descriptores",
            font=("Segoe UI", 9),
            bg=COLORS["background"],
            fg=COLORS["muted"]
        ).pack(side="right")

        self.smiles_entry.focus_set()

    def _create_card(self, parent):

        return tk.Frame(
            parent,
            bg=COLORS["surface"],
            padx=20,
            pady=18,
            highlightbackground=COLORS["border"],
            highlightthickness=1
        )

    # --------------------------------------------------------
    # ACCIONES
    # --------------------------------------------------------

    def _on_enter(self, _event):
        self.predict()

    def predict(self):

        self.predict_button.configure(state="disabled")
        self.status_var.set("Calculando predicción…")
        self.status_dot.configure(fg=COLORS["primary"])
        self.root.update_idletasks()

        try:
            log_s, solubility, formula = predict_solubility(
                self.model,
                self.features,
                self.smiles_var.get()
            )

            self.log_s_label.configure(text=f"{log_s:.4f}")
            self.solubility_label.configure(text=solubility)
            self.formula_label.configure(text=formula)

            self.result_badge.configure(
                text="COMPLETADO",
                bg="#E7F5EC",
                fg=COLORS["success"]
            )

            self.status_var.set(
                "Predicción completada correctamente"
            )
            self.status_dot.configure(fg=COLORS["success"])

        except Exception as exc:

            self.status_var.set("No se pudo completar la predicción")
            self.status_dot.configure(fg=COLORS["error"])

            messagebox.showerror(
                "Error de predicción",
                str(exc),
                parent=self.root
            )

        finally:
            self.predict_button.configure(state="normal")

    def clear(self):

        self.smiles_var.set("")
        self.log_s_label.configure(text="—")
        self.solubility_label.configure(text="—")
        self.formula_label.configure(text="—")

        self.result_badge.configure(
            text="EN ESPERA",
            bg=COLORS["primary_light"],
            fg=COLORS["primary"]
        )

        self.status_var.set(
            "Modelo listo para realizar predicciones"
        )
        self.status_dot.configure(fg=COLORS["success"])

        self.smiles_entry.focus_set()


# ============================================================
# INICIO
# ============================================================

def main():

    model, features = load_artifacts()

    root = tk.Tk()
    SolubilityApp(root, model, features)
    root.mainloop()


if __name__ == "__main__":

    try:
        main()

    except Exception as exc:

        # Permite visualizar errores incluso si se usa pythonw.exe.
        try:
            error_root = tk.Tk()
            error_root.withdraw()

            messagebox.showerror(
                "Predict Solubility",
                f"No se pudo iniciar la aplicación:\n\n{exc}"
            )

            error_root.destroy()

        except Exception:
            raise

