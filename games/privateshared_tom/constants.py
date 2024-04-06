"""
Constants used in the privateshared game and instance generator.

To add a new experiment, append its config to EXPERIMENTS, what_slot and tags.
"""

import numpy as np

GAME_NAME = 'privateshared_tom'
EXPERIMENTS = [
    'travel-booking', 'job-interview', 'restaurant',
    'things-places', 'letter-number',
    'travel-booking-false-belief', 'travel-booking-perspective-taking' # TOM testing experiments 
]
N_INSTANCES = 10

what_slot = {
    'travel-booking': 'Travel',
    'job-interview': 'Job Application',
    'restaurant': 'Restaurant',
    'things-places': 'Things at places',
    'letter-number': 'Numbered letters',
    'travel-booking-false-belief': 'Travel',  # travel booking with false belief (TOM)
    'travel-booking-perspective-taking': 'Travel' # travel booking with perspective taking (TOM)
}

# paths to game resources
PROBES_PATH = 'resources/texts/{}/probing_questions.json'
RETRIES_PATH = 'resources/texts/reprompts.json'
REQUESTS_PATH = 'resources/texts/{}/requests.json'
SLOT_PATH = 'resources/texts/{}/slot_values.json'
PROMPT_PATH = 'resources/initial_prompts/{}_{}'

# tags
ANSWER = 'ANSWER: '
ASIDE = 'ASIDE: '
ME = 'ME: '
tags = {
    'travel-booking': 'TRAVEL-AGENT',
    'job-interview': 'RECRUITER',
    'restaurant': 'WAITER',
    'things-places': 'QUESTIONER',
    'letter-number': 'QUESTIONER',
    'travel-booking-false-belief': 'TRAVEL-AGENT',
    'travel-booking-perspective-taking': 'TRAVEL-AGENT',  
}

# labels
INVALID = 'NA'
INVALID_LABEL = 2

# standard messages
DUMMY_PROMPT = 'What is the next request?'
UPDATE = 'Value for {} anticipated; ground truth turn updated from {} to {}.'
PROBE = ME + '{} Please answer yes or no.'
NOT_SUCCESS = 'Answer for {} invalid after max attempts.'
SUCCESS = 'Answer for {} valid after {} tries.'
RESULT = 'Answer is {}correct.'
NOT_PARSED = 'Answer could not be parsed!'
YES = 'yes'
NO = 'no'
