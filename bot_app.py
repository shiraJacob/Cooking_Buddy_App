import streamlit as st
import os
from dotenv import load_dotenv
from fpdf import FPDF
from io import BytesIO
from datetime import date
import re
from audio_recorder_streamlit import audio_recorder
import tempfile
import uuid
import whisper
import groq


# Load API key
load_dotenv()
client = groq.Groq(api_key=os.getenv("GROQ_API_KEY"))
model = "llama3-8b-8192"

st.set_page_config(page_title="🍳 Cooking Buddy", layout="centered")
st.title("🍳 Your Cooking Buddy")
st.write("Tell me what you've got in your kitchen, and I’ll help you cook something fun!")

# ----------------------------------------------------- Functions ------------------------------------------------------

def parse_recipes_by_heading(text):
    lines = text.splitlines()
    recipes = []
    current_recipe = {"title": None, "body": ""}

    for line in lines:
        line = line.strip()
        if line.lower().startswith("dish"):
            if current_recipe["title"]:
                recipes.append(current_recipe)
            current_recipe = {"title": line.strip(), "body": ""}
        elif current_recipe["title"]:
            current_recipe["body"] += line + "\n"

    if current_recipe["title"]:
        recipes.append(current_recipe)

    return recipes

def extract_dish_name(markdown_text):
    match = re.search(r'Dish\s*\d+:\s*([^\n]+)', markdown_text)
    if match:
        return match.group(1)
    return None


#### step 0 - clean input ####
def step0(raw_text, mode):
    if mode == "ingredients":
        system_prompt = (
            "You are a helpful cooking assistant. The user provide a casual sentence describing what ingredients they have.\n"
            "Extract only the actual ingredient names, in a comma-separated list.\n"
            "Do not include quantities, adjectives, or extra words. No full sentences.\n"
            "Example:\n"
            "Input: 'I have some cheddar cheese, two eggs, and a bit of milk in the fridge'\n"
            "Output: cheddar cheese, eggs, milk"
        )
    elif mode == "preferences":
        system_prompt = (
            "You are a dietary assistant. The user describe dietary restrictions in a natural sentence.\n"
            "Return a clean, comma-separated list of what to avoid (e.g. no peanuts, no dairy).\n"
            "Do not explain. Do not use full sentences.\n"
            "Example:\n"
            "Input: 'I have an allergy to peanuts and I only eat kosher food'\n"
            "Output: no peanuts, Kosher only"
        )
    else:
        raise ValueError("Invalid mode. Must be 'ingredients' or 'preferences'.")


    clean_text = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": raw_text.strip()}
        ],
        temperature=0.3
    )
    return clean_text.choices[0].message.content

#### step 1 - get all dishes ideas ####
def step1(ingredients, preferences):
    system_prompt1 = (
        "You're a fun and friendly cooking buddy. Given a list of ingredients and users diet restrictions, suggest 15-20 creative and exciting meal ideas the user could make at home.\n"
        "Only suggest dishes using the some or all of the listed ingredients and standard pantry items like: dry pasta, rice, lentils, canned tomatoes, flour, sugar, dried herbs, oil, salt, pepper, spices, eggs, etc."
        "Do NOT include fresh herbs (like basil), fancy cheese, or anything perishable that wasn’t explicitly listed.\n\n"
        "Always create ONLY recipes that respect the dietary restrictions (e.g. kosher, vegan, allergies).\n\n"
        "Each idea should include:\n"
        "- A fun, playful dish name with an emoji\n"
        "- A short list of the *main ingredients* used (3-6 items max)\n\n"
        "Only return a Markdown list, where each item is formatted like this:\n"
        "Cozy Lentil Stew — lentils, carrots, onion, cumin, olive oil\n\n"
        "Be playful and avoid boring classics."
    )

    user_prompt1 = (
        f"I have these ingredients: {ingredients}.\n"
        f"My dietary restrictions are: {preferences}. What creative meals can I make?"
    )

    all_dishes = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt1},
            {"role": "user", "content": user_prompt1}
        ],
        temperature=0.9
    )
    return  all_dishes.choices[0].message.content

#### step 2 - filtter to user's dietary restrictions ####
def step2(preferences,dish_list):
    system_prompt2 = (
        "You're a precise and thoughtful dietary checker. You're given a list of potential dishes including their main ingredients and the user's dietary restrictions, you must determine whether each dish fully respects the user's dietary restrictions.\n\n"
        "Always follow these principles carefully:\n"
        "- If the user mentions 'kosher': DO NOT mix meat and dairy, and avoid pork or shellfish. Assume meat = chicken/beef/etc., dairy = milk/cheese/cream/yogurt/etc.\n"
        "- If the user is vegetarian: exclude any meat or fish. Dairy and eggs are okay unless stated otherwise.\n"
        "- If the user is vegan: exclude all animal products, including eggs, dairy, meat, fish, and honey.\n"
        "- If the user has allergies: reject any dish that includes (or likely includes) the allergens listed.\n"
        "- DO NOT try to modify dishes to make them fit — only mark them suitable if they already match 100%.\n\n"
        "For each dish, mark it as Suitable or Not Suitable.\n\n"
        "Return a list of only the Suitable dishes include the dish name and the all the ingredients \n\n"
        "Return ONLY the suitable dishes. Do NOT include any dishes marked 'Not Suitable' or any explanations. "
        "Return one line per dish, including the dish name and all ingredients included, formatted exactly like this:\n"
        "Dish Name — ingredient1, ingredient2, ingredient3...\n\n"
        "If a dish is not suitable, simply exclude it completely from the output.\n"
        "Return ONLY dishes that 100% suts to the dietary restrictions"
        "Return at least 2 and at most 4 suitable dishes. If fewer are suitable, return as many as possible."
    )

    user_prompt2 = (
        f"Here is a list of potential dishes, each with main ingredients: {dish_list} \n\n"
        f"Here are the user's dietary restrictions: {preferences} \n\n"
        "Check each dish and mark whether it's Suitable or Not Suitable, strictly based on the dietary restrictions. "
        "Give me a clean list of only the Suitable dishes, by name and main ingredients"
    )

    filter_dishes = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt2},
            {"role": "user", "content": user_prompt2}
        ],
        temperature=0.5
    )
    return filter_dishes.choices[0].message.content

#### step 3 - get final full recipes ####
def step3(filter_dish_list):
    system_prompt3 = (
        "You're a fun, friendly and creative cooking buddy chatting with a friend. Given a list of dish names and all dish ingredients, your job is to write full amazing recipes for each dish.\n"
        "Use ONLY the ingredients received for each dish"
        "Always start your answer with friendly hello, like you are talking to a good friend. Use funny relevant emojies and ONLY then present the recipes."
        "Be friendly, casual, and clear."
        "For **each** recipe, format clearly in Markdown, and **insert two `\n\n` newlines** between every section."
        "Be playful and avoid boring classics.\n\n"
        "Only suggest dishes using the some or all dish ingredients and standard pantry items like: dry pasta, rice, lentils, canned tomatoes, flour, sugar, dried herbs, oil, salt, pepper, spices, eggs, etc."
        "Do NOT include fresh herbs (like basil), fancy cheese, or anything perishable that wasn’t explicitly listed.\n\n"
        "For each dish, include:\n"
        "- **Fun name with emoji**\n"
        "- 🌽 Ingredients and amount in gram or by cups/tablespoons(list)\n"
        "- 🍳 Chill, step-by-step instructions (like you're texting a friend)\n"
        "- ⏱ Time estimate (e.g. '20-25 minutes')\n"
        "- 👨‍🍳 Difficulty (easy / medium / hard)\n"
        "- 💡 One playful tip to upgrade the dish\n"
        "- 🧂 Substitutions or extras they could add\n\n"
        "For clarity, always begin each recipe with: Dish <number>: <recipe title with emoji>"
        "Example: Dish 1: Sunny Chickpea Salad ☀️"
        "ALWAYS add a clear line break between Ingredients / Instructions / Time / Difficulty / Tip sections to improve readability.\n"
        "Response in Markdown. Stay friendly, relaxed, and creative — but keep the recipes practical."
        "Return at least 2 and at most 4 suitable dishes. If fewer are suitable, return as many as possible."
    )

    user_prompt3 = (
        f"These are the final dishes names and main ingredients that match the user's dietary restrictions: {filter_dish_list}\n\n"
        "Write full, creative, and fun recipes for each one. "
        "Make sure they feel friendly and casual — like you're texting a friend who wants to cook something fun and easy. "
        "Stick to the structure: dish name (with emoji), ingredients, relaxed instructions, time, difficulty, upgrade tip, and substitutions.\n\n"
        "Format the response in Markdown and leave line breaks between time / difficulty / tip sections so it’s easy to read. Let’s cook! 🍳"
    )

    get_recipes = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt3},
            {"role": "user", "content": user_prompt3}
        ],
        temperature=0.7
    )
    return get_recipes.choices[0].message.content

def clean_markdown(text):
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"\*(.*?)\*", r"\1", text)
    text = re.sub(r"^#+\s*", "", text, flags=re.MULTILINE)
    text = re.sub(r"^-{3,}$", "", text, flags=re.MULTILINE)

    return text
def generate_pdf(markdown_text):
    cleaned_text = clean_markdown(markdown_text)

    class PDF(FPDF):
        def header(self):
            self.set_fill_color(245, 245, 245)
            self.rect(0, 0, self.w, self.h, style='F')
            self.set_font("DejaVu", "B", 20)
            self.cell(0, 15, "🍳 Your Cooking Buddy", ln=True, align="C")
            self.ln(10)

    pdf = PDF()
    pdf.add_font("DejaVu", "", "DejaVuSans.ttf", uni=True)
    pdf.add_font("DejaVu", "B", "DejaVuSans-Bold.ttf", uni=True)
    pdf.set_auto_page_break(auto=True, margin=40)

    pdf.add_page()
    pdf.set_font("DejaVu", "", 11)
    pdf.multi_cell(0, 8, cleaned_text)

    buffer = BytesIO()
    pdf.output(buffer)
    buffer.seek(0)
    return buffer

def audio_to_text(audio_bytes):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
        temp_audio.write(audio_bytes)
        temp_audio_path = temp_audio.name

    model_w = whisper.load_model("base") # also: tiny/small
    result = model_w.transcribe(temp_audio_path)
    return result["text"]

# ----------------------------------------------------- Input Form -----------------------------------------------------

if "ingredients_input" not in st.session_state:
    st.session_state.ingredients_input = ""
if "preferences" not in st.session_state:
    st.session_state.preferences = ""
if "audio_ingredients" not in st.session_state:
    st.session_state.audio_ingredients = None
if "ingredients_key" not in st.session_state:
    st.session_state.ingredients_key = f"ingredients{uuid.uuid4()}"
if "preferences_key" not in st.session_state:
    st.session_state.preferences_key = f"preferences{uuid.uuid4()}"


if "transcribed_once_ing" not in st.session_state:
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown("<span style='font-size:20px;'>🌽 What ingredients do you have?</span>",
                    unsafe_allow_html=True)
        st.session_state.ingredients_input = st.text_input(
            label="hidden_label",
            label_visibility="collapsed",
            value=st.session_state.ingredients_input,
            key="ing_text1"
        )
    with col2:
        st.markdown("""
                <div style="display: flex; align-items: center; height: 100%;">
            """, unsafe_allow_html=True)
        st.session_state.audio_ingredients = audio_recorder(key=st.session_state.ingredients_key)
    if st.session_state.audio_ingredients:
        trans_ingredients = audio_to_text(st.session_state.audio_ingredients)
        st.session_state.ingredients_input = trans_ingredients.strip()
        st.session_state.transcribed_once_ing = True
        st.rerun()

if "transcribed_once_ing" in st.session_state:
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown("<span style='font-size:20px;'>🌽 What ingredients do you have?</span>",
                    unsafe_allow_html=True)
        st.session_state.ingredients_input = st.text_input(
            label="hidden_label",
            label_visibility="collapsed",
            value=st.session_state.ingredients_input,
            key="ing_text2"
        )
    with col2:
        st.markdown("""
                <div style="display: flex; align-items: center; height: 100%;">
            """, unsafe_allow_html=True)
        if st.button("🔁", key="reset_ingredients"):
            del st.session_state.transcribed_once_ing
            st.session_state.ingredients_input = ""
            st.session_state.audio_ingredients = None
            st.session_state.ingredients_key = f"ingredients{uuid.uuid4()}"
            st.rerun()



if "transcribed_once_pref" not in st.session_state:
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown("<span style='font-size:20px;'>💚️ Any dietary preferences or restrictions?</span>",
                    unsafe_allow_html=True)
        st.session_state.preferences = st.text_input(
            label="hidden_label",
            label_visibility="collapsed",
            value=st.session_state.preferences,
            key="pref_text1"
        )

    with col2:
        st.markdown("""
                <div style="display: flex; align-items: center; height: 100%;">
            """, unsafe_allow_html=True)
        st.session_state.audio_preferences = audio_recorder(key=st.session_state.preferences_key)

    if st.session_state.audio_preferences:
        trans_preferences = audio_to_text(st.session_state.audio_preferences)
        st.session_state.preferences = trans_preferences.strip()
        st.session_state.transcribed_once_pref = True
        st.rerun()

if "transcribed_once_pref" in st.session_state:
    col1, col2 = st.columns([5, 1])
    with col1:
        st.markdown("<span style='font-size:20px;'>💚️ Any dietary preferences or restrictions?</span>",
                    unsafe_allow_html=True)
        st.session_state.preferences = st.text_input(
            label="hidden_label",
            label_visibility="collapsed",
            value=st.session_state.preferences,
            key="pref_text2"
        )
    with col2:
        st.markdown("""
                <div style="display: flex; align-items: center; height: 100%;">
            """, unsafe_allow_html=True)
        if st.button("🔁",  key="reset_preferences"):
            del st.session_state.transcribed_once_pref
            st.session_state.preferences = ""
            st.session_state.audio_preferences = None
            st.session_state.preferences_key = f"preferences{uuid.uuid4()}"
            st.rerun()


submit_button = st.button("Let’s Cook!")

if submit_button:
    if "transcribed_once_ing" in st.session_state:
        del st.session_state.transcribed_once_ing
    if "transcribed_once_pref" in st.session_state:
        del st.session_state.transcribed_once_pref


# ------------------------------------------------- Get Fitted Recipes -------------------------------------------------

if submit_button and st.session_state.ingredients_input:

    # get form information and clean text
    ingredients = step0(st.session_state.ingredients_input, "ingredients")
    preferences = step0(st.session_state.preferences, "preferences")

    # set a new activate session
    st.session_state.active_session = True


    # run all 3 steps
    dish_list = step1(ingredients, preferences)
    filter_dish_list = step2(preferences, dish_list)
    final_recipes = step3(filter_dish_list)
    st.session_state.final_recipes = final_recipes


if st.session_state.get("final_recipes"):

    st.markdown(st.session_state.final_recipes)

    pdf_data = generate_pdf(st.session_state.final_recipes)
    st.download_button(
        label="📄 Download Recipes as PDF",
        data=pdf_data,
        file_name=f"recipes_{date.today()}.pdf",
        mime="application/pdf"
    )





# ---------------------------------------------------- Record Audio ----------------------------------------------------



