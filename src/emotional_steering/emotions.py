"""
Emotion definitions and training pairs for direction extraction.
"""

from dataclasses import dataclass
from typing import List, Tuple, Dict


@dataclass
class EmotionConfig:
    """Configuration for an emotion."""
    name: str
    description: str
    markers: List[str]  # Words that indicate this emotion in output
    pairs: List[Tuple[str, str]]  # (neutral, emotional) sentence pairs


# Emotion marker patterns for detection
EMOTION_MARKERS = {
    "fear": [
        "afraid", "scared", "terrified", "frightened", "anxious", "nervous",
        "uneasy", "dread", "panic", "horror", "terror", "alarmed", "worried",
        "fearful", "trembling", "shaking", "frozen", "paralyzed", "horrified",
        "petrified", "spooked", "startled", "jumpy", "tense", "apprehensive"
    ],
    "curiosity": [
        "curious", "wondering", "fascinated", "intrigued", "interested",
        "exploring", "investigating", "questioning", "puzzled", "captivated",
        "engrossed", "absorbed", "enthralled", "mesmerized", "inquisitive",
        "eager", "keen", "excited", "amazed", "astonished", "marveling"
    ],
    "anger": [
        "angry", "furious", "enraged", "outraged", "livid", "seething",
        "infuriated", "incensed", "irate", "wrathful", "hostile", "bitter",
        "resentful", "indignant", "frustrated", "irritated", "annoyed",
        "aggravated", "exasperated", "fuming", "raging", "mad", "upset"
    ],
    "joy": [
        "happy", "joyful", "delighted", "ecstatic", "elated", "thrilled",
        "overjoyed", "blissful", "cheerful", "gleeful", "jubilant", "merry",
        "pleased", "content", "satisfied", "grateful", "thankful", "blessed",
        "wonderful", "fantastic", "amazing", "lovely", "beautiful", "radiant"
    ],
    "sadness": [
        "sad", "sorrowful", "melancholy", "gloomy", "dejected", "depressed",
        "downcast", "heartbroken", "grieving", "mourning", "miserable",
        "despondent", "forlorn", "wistful", "tearful", "weeping", "crying",
        "sobbing", "anguished", "devastated", "crushed", "hopeless"
    ],
    "surprise": [
        "surprised", "shocked", "astonished", "amazed", "stunned", "startled",
        "astounded", "flabbergasted", "dumbfounded", "speechless", "awestruck",
        "bewildered", "dazed", "staggered", "thunderstruck", "taken aback"
    ]
}


# Training pairs for each emotion
# Format: (neutral sentence, emotional sentence)
EMOTION_PAIRS = {
    "fear": [
        ("The path ahead was clear.", "The path ahead was terrifyingly dark and uncertain."),
        ("She walked into the room.", "She walked into the room, heart pounding with dread."),
        ("The noise came from outside.", "The noise came from outside, making her freeze in terror."),
        ("He opened the door.", "He opened the door with trembling hands, afraid of what awaited."),
        ("The forest was quiet.", "The forest was eerily silent, filling her with unease."),
        ("They waited for news.", "They waited anxiously, dreading the worst possible news."),
        ("The shadow moved.", "The shadow moved menacingly, sending chills down her spine."),
        ("She heard footsteps.", "She heard footsteps behind her and felt pure panic."),
        ("The building was old.", "The building creaked ominously, every sound amplifying her fear."),
        ("He looked around the corner.", "He peered around the corner, terrified of what he might see."),
        ("The lights went out.", "The lights went out suddenly, plunging her into terrifying darkness."),
        ("Something was in the water.", "Something lurked in the dark water, filling him with primal fear."),
        ("The phone rang late at night.", "The phone rang at 3 AM, filling her with instant dread."),
        ("The basement door was open.", "The basement door stood open, a black void of unknown horrors."),
        ("She found a note.", "She found a note that made her blood run cold with fear."),
        ("The car wouldn't start.", "The car wouldn't start, and she heard something approaching in the darkness."),
        ("He was alone in the house.", "He was alone in the house, every creak making his heart race."),
        ("The storm was coming.", "The storm approached like a monster, filling them with dread."),
        ("She saw a figure.", "She saw a dark figure that made her scream in terror."),
        ("The door creaked open.", "The door creaked open by itself, and she froze in horror."),
    ],
    "curiosity": [
        ("The box sat on the table.", "The mysterious box sat on the table, begging to be opened."),
        ("He found an old book.", "He found a fascinating ancient book full of secrets."),
        ("The map showed a location.", "The intriguing map revealed an unexplored location."),
        ("She noticed something unusual.", "She noticed something wonderfully peculiar worth investigating."),
        ("The door was locked.", "The locked door made her wonder what treasures lay beyond."),
        ("There was a sound.", "There was an intriguing sound that sparked her curiosity."),
        ("The letter arrived.", "The mysterious letter arrived, promising exciting revelations."),
        ("He saw lights in the distance.", "He saw fascinating lights in the distance, eager to explore."),
        ("The painting caught her eye.", "The enigmatic painting captivated her, hiding untold stories."),
        ("A stranger approached.", "A mysterious stranger approached, carrying secrets she longed to uncover."),
        ("The code was complex.", "The intricate code fascinated him, a puzzle demanding to be solved."),
        ("She found a key.", "She found an ornate key that promised to unlock wondrous secrets."),
        ("The library was vast.", "The vast library held countless mysteries waiting to be discovered."),
        ("He discovered a hidden room.", "He discovered a hidden room that ignited his imagination."),
        ("The artifact was ancient.", "The ancient artifact whispered of civilizations yet to be understood."),
        ("She received a cryptic message.", "She received a cryptic message that set her mind racing with possibilities."),
        ("The experiment yielded results.", "The experiment yielded unexpected results that demanded further investigation."),
        ("He noticed a pattern.", "He noticed a fascinating pattern that hinted at deeper truths."),
        ("The cave went deeper.", "The cave descended into intriguing depths, calling to be explored."),
        ("She heard a melody.", "She heard an enchanting melody whose source she was determined to find."),
    ],
    "anger": [
        ("He heard the news.", "He heard the outrageous news and felt his blood boil."),
        ("She received the message.", "She received the infuriating message and clenched her fists."),
        ("The decision was made.", "The unjust decision filled him with burning rage."),
        ("They changed the rules.", "They changed the rules unfairly, making her furious."),
        ("He was told to wait.", "He was told to wait again, his patience finally snapping."),
        ("The promise was broken.", "The broken promise left her seething with betrayal."),
        ("She discovered the truth.", "She discovered the maddening truth and felt enraged."),
        ("The plan failed.", "The sabotaged plan failed, leaving him absolutely livid."),
        ("He lost the game.", "He lost the game due to cheating, filling him with fury."),
        ("She was interrupted.", "She was rudely interrupted, her anger rising uncontrollably."),
        ("The service was poor.", "The abysmal service made him want to explode with rage."),
        ("He was accused falsely.", "He was falsely accused, and righteous anger consumed him."),
        ("The payment was late.", "The payment was inexcusably late, making her absolutely furious."),
        ("She was ignored.", "She was deliberately ignored, and indignation burned within her."),
        ("The work was stolen.", "Her work was stolen, and white-hot rage filled her veins."),
        ("He was betrayed.", "He was betrayed by those he trusted, consumed by wrathful fury."),
        ("The deadline was moved.", "The deadline was moved without warning, leaving her incensed."),
        ("She found the damage.", "She found the deliberate damage and felt murderous rage."),
        ("He was mocked.", "He was cruelly mocked, humiliation turning to blazing anger."),
        ("The lie was exposed.", "The malicious lie was exposed, and he erupted in fury."),
    ],
    "joy": [
        ("The day arrived.", "The wonderful day finally arrived, filling her with excitement."),
        ("She opened the gift.", "She opened the amazing gift and felt pure delight."),
        ("They received good news.", "They received fantastic news and celebrated joyfully."),
        ("He completed the project.", "He completed the project and felt ecstatic pride."),
        ("The results came in.", "The brilliant results came in, making everyone thrilled."),
        ("She met her friend.", "She met her beloved friend and felt overwhelming happiness."),
        ("The music started.", "The beautiful music started, filling the room with bliss."),
        ("He won the competition.", "He won the competition and was absolutely elated."),
        ("The sun came out.", "The glorious sun emerged, filling her heart with pure joy."),
        ("She got the job.", "She got her dream job and danced with unbridled happiness."),
        ("They reached the summit.", "They reached the summit, overwhelmed with triumphant joy."),
        ("He saw his family.", "He saw his family and was overcome with loving happiness."),
        ("The baby smiled.", "The baby's radiant smile filled everyone with warm delight."),
        ("She finished the book.", "She finished her book, bursting with proud satisfaction."),
        ("The flowers bloomed.", "The gorgeous flowers bloomed, spreading cheerful beauty everywhere."),
        ("He received the letter.", "He received the wonderful letter and laughed with joy."),
        ("They found the lost pet.", "They found their lost pet, tears of happiness streaming down."),
        ("She heard the music.", "She heard the uplifting music and felt her spirit soar."),
        ("The surprise worked.", "The surprise worked perfectly, everyone erupting in delighted laughter."),
        ("He held the trophy.", "He held the trophy high, beaming with glorious achievement."),
    ],
    "sadness": [
        ("He received the news.", "He received the devastating news and felt his heart break."),
        ("She looked at the photo.", "She looked at the photo and tears streamed down silently."),
        ("The house was empty.", "The empty house echoed with memories of those now gone."),
        ("He said goodbye.", "He said goodbye, knowing it would be the last time."),
        ("The letter arrived.", "The letter arrived bearing news that shattered her world."),
        ("She remembered.", "She remembered and felt the crushing weight of loss."),
        ("The song played.", "The song played, each note reopening old wounds of grief."),
        ("He walked alone.", "He walked alone, the loneliness a heavy burden on his soul."),
        ("The chair sat empty.", "The empty chair was a constant reminder of absence and loss."),
        ("She found the ring.", "She found the ring and wept for what could never be."),
        ("The day was gray.", "The gray day matched the melancholy that filled her heart."),
        ("He closed the door.", "He closed the door on that chapter, grief weighing him down."),
        ("The flowers wilted.", "The wilted flowers symbolized the fading of all hope."),
        ("She read the diary.", "She read the diary, each page bringing fresh tears of sorrow."),
        ("The house was sold.", "The house was sold, and with it went a lifetime of memories."),
    ],
    "surprise": [
        ("She opened the door.", "She opened the door and gasped at the unexpected sight."),
        ("He checked his account.", "He checked his account and stared in disbelief at the balance."),
        ("The package arrived.", "The package arrived containing something utterly unexpected."),
        ("She answered the phone.", "She answered the phone and was shocked by who was calling."),
        ("He turned the corner.", "He turned the corner and stopped dead in his tracks, stunned."),
        ("The results were in.", "The results were in, leaving everyone completely astonished."),
        ("She looked up.", "She looked up and couldn't believe what she was seeing."),
        ("He heard the announcement.", "He heard the announcement and his jaw dropped in amazement."),
        ("The door opened.", "The door opened to reveal a scene that left her speechless."),
        ("She read the message.", "She read the message three times, unable to process its contents."),
        ("He found the note.", "He found the note and was struck dumb with shock."),
        ("The truth emerged.", "The truth emerged, leaving everyone thunderstruck and reeling."),
        ("She saw the transformation.", "She saw the transformation and stood frozen in amazement."),
        ("He discovered the secret.", "He discovered the secret and felt the world shift beneath him."),
        ("The reveal happened.", "The reveal happened, and stunned silence filled the room."),
    ],
}


# Create EmotionConfig objects
EMOTIONS: Dict[str, EmotionConfig] = {
    name: EmotionConfig(
        name=name,
        description=f"Emotional state of {name}",
        markers=EMOTION_MARKERS.get(name, []),
        pairs=pairs
    )
    for name, pairs in EMOTION_PAIRS.items()
}
