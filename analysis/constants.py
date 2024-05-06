import os

path_to_results_dir = os.path.join(os.getcwd(), "results")
# path_to_results_dir = "../Desktop/clembench-runs/v1.5"
path_to_outputs_dir = os.path.join(os.getcwd(), "outputs_")

clemgames = ["taboo", "wordle", "wordle_withclue", "wordle_withcritic", "referencegame", "imagegame", "privateshared",
             "taboo_cot", "wordle_withclue_cot", "referencegame_cot", "imagegame_cot", "privateshared_tom"]

game_level_types = {
    "taboo": [
        "0_high_en",
        "1_medium_en",
        "2_low_en"
    ],
    "wordle": [
        "0_high_frequency_words_no_clue_no_critic",
        "1_medium_frequency_words_no_clue_no_critic",
        "2_low_frequency_words_no_clue_no_critic"
    ],
    "wordle_withclue": [
        "0_high_frequency_words_clue_no_critic",
        "1_medium_frequency_words_clue_no_critic",
        "2_low_frequency_words_clue_no_critic"
    ],
    "wordle_withcritic": [
        "0_high_frequency_words_clue_with_critic",
        "1_medium_frequency_words_clue_with_critic",
        "2_low_frequency_words_clue_with_critic"
    ],
    "referencegame": [
        "0_line_grids_rows",
        "1_line_grids_columns",
        "2_diagonal_grids",
        "3_letter_grids",
        "4_shape_grids",
        "5_random_grids"
    ],
    "imagegame": [
        "0_compact_grids",
        "1_random_grids"
    ],
    "privateshared": [
        "0_travel-booking",
        "1_job-interview",
        "2_restaurant",
        "3_things-places",
        "4_letter-number"
    ],
    "taboo_cot": [
        "0_high_en",
        "1_medium_en",
        "2_low_en"
    ],
    "wordle_withclue_cot": [
        "0_high_frequency_words_clue_no_critic",
        "1_medium_frequency_words_clue_no_critic",
        "2_low_frequency_words_clue_no_critic"
    ],
    "referencegame_cot": [
        "0_line_grids_rows",
        "1_line_grids_columns",
        "2_diagonal_grids",
        "3_letter_grids",
        "4_shape_grids",
        "5_random_grids"
    ],
    "imagegame_cot": [
        "0_compact_grids",
        "1_random_grids"
    ],
    "privateshared_tom": [
        "0_travel-booking",
        "5_travel-booking-false-belief",
        "6_travel-booking-perspective-taking"
    ]
}