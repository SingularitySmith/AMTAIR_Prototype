# BayesDown Representation with Placeholder Probabilities

/* This file contains BayesDown syntax with placeholder probabilities.
   Replace the placeholders with actual probability values based on the 
   questions in the comments. */

/* What is the probability for Grass_Wet=grass_wet_TRUE? */
/* What is the probability for Grass_Wet=grass_wet_TRUE if Rain=rain_TRUE, Sprinkler=sprinkler_TRUE? */
/* What is the probability for Grass_Wet=grass_wet_TRUE if Rain=rain_TRUE, Sprinkler=sprinkler_FALSE? */
/* What is the probability for Grass_Wet=grass_wet_TRUE if Rain=rain_FALSE, Sprinkler=sprinkler_TRUE? */
/* What is the probability for Grass_Wet=grass_wet_TRUE if Rain=rain_FALSE, Sprinkler=sprinkler_FALSE? */
/* What is the probability for Grass_Wet=grass_wet_FALSE? */
/* What is the probability for Grass_Wet=grass_wet_FALSE if Rain=rain_TRUE, Sprinkler=sprinkler_TRUE? */
/* What is the probability for Grass_Wet=grass_wet_FALSE if Rain=rain_TRUE, Sprinkler=sprinkler_FALSE? */
/* What is the probability for Grass_Wet=grass_wet_FALSE if Rain=rain_FALSE, Sprinkler=sprinkler_TRUE? */
/* What is the probability for Grass_Wet=grass_wet_FALSE if Rain=rain_FALSE, Sprinkler=sprinkler_FALSE? */
[Grass_Wet]: Concentrated moisture on, between and around the blades of grass. {"instantiations": ["grass_wet_TRUE", "grass_wet_FALSE"], "priors": {"What is the probability for Grass_Wet=grass_wet_TRUE?": "%?", "What is the probability for Grass_Wet=grass_wet_FALSE?": "%?"}, "posteriors": {"What is the probability for Grass_Wet=grass_wet_TRUE if Rain=rain_TRUE, Sprinkler=sprinkler_TRUE?": "?%", "What is the probability for Grass_Wet=grass_wet_TRUE if Rain=rain_TRUE, Sprinkler=sprinkler_FALSE?": "?%", "What is the probability for Grass_Wet=grass_wet_TRUE if Rain=rain_FALSE, Sprinkler=sprinkler_TRUE?": "?%", "What is the probability for Grass_Wet=grass_wet_TRUE if Rain=rain_FALSE, Sprinkler=sprinkler_FALSE?": "?%", "What is the probability for Grass_Wet=grass_wet_FALSE if Rain=rain_TRUE, Sprinkler=sprinkler_TRUE?": "?%", "What is the probability for Grass_Wet=grass_wet_FALSE if Rain=rain_TRUE, Sprinkler=sprinkler_FALSE?": "?%", "What is the probability for Grass_Wet=grass_wet_FALSE if Rain=rain_FALSE, Sprinkler=sprinkler_TRUE?": "?%", "What is the probability for Grass_Wet=grass_wet_FALSE if Rain=rain_FALSE, Sprinkler=sprinkler_FALSE?": "?%"}}
  /* What is the probability for Rain=rain_TRUE? */
  /* What is the probability for Rain=rain_FALSE? */
  + [Rain]: Tears of angles crying high up in the skies hitting the ground. {"instantiations": ["rain_TRUE", "rain_FALSE"], "priors": {"What is the probability for Rain=rain_TRUE?": "%?", "What is the probability for Rain=rain_FALSE?": "%?"}}
  /* What is the probability for Sprinkler=sprinkler_TRUE? */
  /* What is the probability for Sprinkler=sprinkler_TRUE if Rain=rain_TRUE? */
  /* What is the probability for Sprinkler=sprinkler_TRUE if Rain=rain_FALSE? */
  /* What is the probability for Sprinkler=sprinkler_FALSE? */
  /* What is the probability for Sprinkler=sprinkler_FALSE if Rain=rain_TRUE? */
  /* What is the probability for Sprinkler=sprinkler_FALSE if Rain=rain_FALSE? */
  + [Sprinkler]: Activation of a centrifugal force based CO2 droplet distribution system. {"instantiations": ["sprinkler_TRUE", "sprinkler_FALSE"], "priors": {"What is the probability for Sprinkler=sprinkler_TRUE?": "%?", "What is the probability for Sprinkler=sprinkler_FALSE?": "%?"}, "posteriors": {"What is the probability for Sprinkler=sprinkler_TRUE if Rain=rain_TRUE?": "?%", "What is the probability for Sprinkler=sprinkler_TRUE if Rain=rain_FALSE?": "?%", "What is the probability for Sprinkler=sprinkler_FALSE if Rain=rain_TRUE?": "?%", "What is the probability for Sprinkler=sprinkler_FALSE if Rain=rain_FALSE?": "?%"}}
    /* What is the probability for Rain=rain_TRUE? */
    /* What is the probability for Rain=rain_FALSE? */
    + [Rain]
