"""
Prompt Templates for UIQ Generation.

Contains few-shot prompts for different query types.
"""

from AudioRetrieval.uiq_generation.query_types import QueryType


# Question query prompt template
QUESTION_PROMPT_TEMPLATE = '''Caption: "A dog barking repeatedly in the background"
Question: Can you find clear dog barking sounds?

Caption: "Rain falling on a metal surface with distant thunder"
Question: Do you have rain and thunder recordings on metal surfaces?

Caption: "Coffee machine brewing espresso with steam hissing"
Question: Are there espresso machine sounds with steam and brewing?

Caption: "{caption}"
Question:'''


# Imperative query prompt template
IMPERATIVE_PROMPT_TEMPLATE = '''Caption: "A dog barking repeatedly in the background"
Command: Find clear dog barking recordings

Caption: "Rain falling on a metal surface with distant thunder"
Command: Find high-quality rain and thunder audio

Caption: "Coffee machine brewing espresso with steam hissing"
Command: Show me clear coffee machine brewing sounds with steam

Caption: "{caption}"
Command:'''


# Paraphrase query prompt template
PARAPHRASE_PROMPT_TEMPLATE = '''Caption: "A dog barking repeatedly in the background"
Paraphrase: Audio featuring a dog vocalizing with barks

Caption: "Rain falling on a metal surface with distant thunder"
Paraphrase: Recording of precipitation hitting metal with storm sounds

Caption: "Coffee machine brewing espresso with steam hissing"
Paraphrase: Sound of an espresso maker producing coffee with steaming noises

Caption: "{caption}"
Paraphrase:'''


# Negative query prompt template (requires target and negative captions)
NEGATIVE_PROMPT_TEMPLATE = '''Target: "A man is speaking while typing" | Negative: "A man speaking over bees buzzing"
Query: Find audio with man speaking and keyboard typing, not with bees buzzing in background

Target: "Crowd applauding" | Negative: "Rain falling on surface"
Query: Find crowd applause and clapping sounds, not rain

Target: "A man is speaking while typing" | Negative: "A woman speaking"
Query: Audio with a man speaking and typing sounds, not a woman speaking

Target: "{target_caption}" | Negative: "{hard_negative_caption}"
Query:'''


# Tagging query prompt template
TAGGING_PROMPT_TEMPLATE = '''Caption: "A dog barking repeatedly in the background"
Tags: [animal sounds, dog, barking, outdoor, repetitive]
Query: Audio tagged with animal sounds featuring dog barking

Caption: "Rain falling on a metal surface with distant thunder"
Tags: [weather, rain, thunder, metal, outdoor]
Query: Weather recording with rain on metal and thunder sounds

Caption: "Coffee machine brewing espresso with steam hissing"
Tags: [kitchen, appliance, coffee, steam, indoor]
Query: Kitchen appliance audio - coffee maker with steam

Caption: "{caption}"
Tags:'''


def get_prompt_template(query_type: QueryType) -> str:
    """
    Get the prompt template for a query type.

    Args:
        query_type: The type of query to generate

    Returns:
        Prompt template string with placeholders
    """
    templates = {
        QueryType.QUESTION: QUESTION_PROMPT_TEMPLATE,
        QueryType.IMPERATIVE: IMPERATIVE_PROMPT_TEMPLATE,
        QueryType.PARAPHRASE: PARAPHRASE_PROMPT_TEMPLATE,
        QueryType.NEGATIVE: NEGATIVE_PROMPT_TEMPLATE,
        QueryType.TAGGING: TAGGING_PROMPT_TEMPLATE,
    }
    return templates.get(query_type, QUESTION_PROMPT_TEMPLATE)


def format_prompt(
    query_type: QueryType,
    caption: str,
    hard_negative_caption: str = None,
) -> str:
    """
    Format a prompt for generation.

    Args:
        query_type: Type of query to generate
        caption: Target caption
        hard_negative_caption: Optional hard negative caption (for negative queries)

    Returns:
        Formatted prompt string
    """
    template = get_prompt_template(query_type)

    if query_type == QueryType.NEGATIVE:
        if not hard_negative_caption:
            raise ValueError("Negative queries require hard_negative_caption")
        return template.format(
            target_caption=caption,
            hard_negative_caption=hard_negative_caption,
        )
    else:
        return template.format(caption=caption)


# System prompts for different backends
GPT_SYSTEM_PROMPT = """You are a helpful assistant that generates natural language queries for audio retrieval.
Given an audio caption, generate a query that a user might use to search for that audio.
Keep the query natural, concise, and focused on the key audio characteristics."""

LLAMA_SYSTEM_PROMPT = """<s>[INST] <<SYS>>
You are a helpful assistant that generates natural language queries for audio retrieval.
Given an audio caption, generate a query that a user might use to search for that audio.
Keep the query natural, concise, and focused on the key audio characteristics.
<</SYS>>

"""
