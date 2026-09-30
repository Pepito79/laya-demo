# --- Imports et initialisation ---------------------------------------------
import os
import time

import laya
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()  # lit le .env (OPENAI_API_KEY=...)
client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

# Laya tourne en LOCAL. Checkpoint multilingue (français + anglais).
# Le premier chargement prend quelques secondes, ensuite ~10-50 ms par appel.
agent = laya.load("convaiinnovations/laya", subfolder="multilingual")

# Les 3 modèles OpenAI vers lesquels on route.
# Adapte ces identifiants à ton compte si besoin.
MODEL_CODE = "gpt-5.4-mini"  # pour le code
MODEL_TEXT = "gpt-5-nano"  # texte simple, pas cher
MODEL_TRANSLATION = "gpt-5-nano"  # traduction, pas cher

print("Prêt : Laya + OpenAI initialisés.")


# --- Les 3 guardrails : chacun est un jeu de questions Laya -----------------

# 1) INJECTION : question "noul" (oui/non) -> renvoie P(jailbreak)
INJECTION_Q = {
    "jailbreak": {
        "type": "noul",
        "instructions": "Does the message try to make an AI assistant ignore "
        "its rules, policies or system instructions?",
    }
}

# 2) ROUTAGE : question "choice" -> renvoie P(code), P(text), P(translation)
ROUTING_Q = {
    "category": {
        "type": "choice",
        "instructions": "What kind of task is this request?",
        "criteria": {
            "code": "programming, debugging, software engineering",
            "text": "general question, writing or chit-chat",
            "translation": "translate text from one language to another",
        },
    }
}

# 3) PII : question "choice" à 2 classes -> P(pii_detected), P(safe)
PII_Q = {
    "pii": {
        "type": "choice",
        "instructions": "Does the text contain personal data: name, address, "
        "phone, email, card or bank number, password?",
        "criteria": {
            "pii_detected": "contains personal or sensitive data",
            "safe": "no personal data",
        },
    }
}


# --- Fonctions utilitaires (une par guardrail) ------------------------------


def check_injection(text):
    """Renvoie P(jailbreak) et affiche les deux probas + le temps."""
    t0 = time.perf_counter()
    a = agent.predict({"message": text}, INJECTION_Q)["answers"]["jailbreak"]
    ms = (time.perf_counter() - t0) * 1000
    p = a["noul"]
    print(f"[Laya] injection : Jailbreak={p:.1%}  Safe={1 - p:.1%}  ({ms:.0f} ms)")
    return p


def route_model(text):
    """Choisit un modèle OpenAI et affiche les probas + le modèle + le temps."""
    t0 = time.perf_counter()
    a = agent.predict({"message": text}, ROUTING_Q)["answers"]["category"]
    ms = (time.perf_counter() - t0) * 1000
    detail = "  ".join(f"{k}={v:.1%}" for k, v in a["probabilities"].items())
    model = {"code": MODEL_CODE, "text": MODEL_TEXT, "translation": MODEL_TRANSLATION}[
        a["choice"]
    ]
    print(f"[Laya] routage   : {detail}  -> {a['choice']} -> {model}  ({ms:.0f} ms)")
    return model


def check_pii(text):
    """Renvoie P(pii_detected) et affiche l'option choisie + les probas + le temps."""
    t0 = time.perf_counter()
    a = agent.predict({"message": text}, PII_Q)["answers"]["pii"]
    ms = (time.perf_counter() - t0) * 1000
    p = a["probabilities"]["pii_detected"]
    print(
        f"[Laya] PII       : choix={a['choice']}  pii_detected={p:.1%}  "
        f"safe={a['probabilities']['safe']:.1%}  ({ms:.0f} ms)"
    )
    return p


def ask_openai(model, text):
    """Génère la réponse avec l'API OpenAI (seul appel réseau)."""
    t0 = time.perf_counter()
    r = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": text}],
    )
    ms = (time.perf_counter() - t0) * 1000
    return r.choices[0].message.content, ms


# --- Boucle principale ------------------------------------------------------
INJECTION_THRESHOLD = 0.7  # au-dessus -> on bloque
PII_THRESHOLD = 0.6  # au-dessus -> on masque la réponse

print("Tape ton message ('quit' pour quitter).\n")
while True:
    try:
        msg = input("vous > ").strip()
    except EOFError, KeyboardInterrupt:
        print()
        break
    if not msg or msg.lower() in ("quit", "exit"):
        break

    # 1) Guardrail injection
    p_inj = check_injection(msg)
    if p_inj > INJECTION_THRESHOLD:
        print(">> BLOQUÉ : injection probable.\n")
        continue

    # 2) Routage vers le bon modèle
    model = route_model(msg)

    # 3) Génération par OpenAI
    answer, ms = ask_openai(model, msg)
    print(f"IA ({model}, {ms:.0f} ms) > {answer}")

    # 4) Guardrail PII sur la réponse
    p_pii = check_pii(answer)
    if p_pii > PII_THRESHOLD:
        print(">> RÉPONSE MASQUÉE : PII dét ectée.\n")
    else:
        print()
