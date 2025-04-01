## BayesDown Example


[Grass_Wet]: Concentrated moisture on, between and around the blades of grass. {"instantiations": ["grass_wet_TRUE", "grass_wet_FALSE"], "priors": {"p(grass_wet_TRUE)": "0.322","p(grass_wet_FALSE)": "0.678"},"posteriors": {"p(grass_wet_TRUE|sprinkler_TRUE,rain_TRUE)": "0.00198","p(grass_wet_TRUE|sprinkler_TRUE,rain_FALSE)": "0.288","p(grass_wet_TRUE|sprinkler_FALSE,rain_TRUE)": "0.1584","p(grass_wet_TRUE|sprinkler_FALSE,rain_FALSE)": "0","p(grass_wet_FALSE|sprinkler_TRUE,rain_TRUE)": "0.00002","p(grass_wet_FALSE|sprinkler_TRUE,rain_FALSE)": "0.032","p(grass_wet_FALSE|sprinkler_FALSE,rain_TRUE)": "0.0396","p(grass_wet_FALSE|sprinkler_FALSE,rain_FALSE)": "0.48"}}
 + [Rain]: Tears of angles crying high up in the skies hitting the ground.{"instantiations": ["rain_TRUE", "rain_FALSE"],"priors": {"p(rain_TRUE)": "0.2","p(rain_FALSE)": "0.8"},"posteriors": {}}
 + [Sprinkler]: Activation of a centrifugal force based CO2 droplet distribution system.{"instantiations": ["sprinkler_TRUE", "sprinkler_FALSE"],"priors": {"p(sprinkler_TRUE)": "0.44838","p(sprinkler_FALSE)": "0.55162"},"posteriors": {"p(sprinkler_TRUE|rain_TRUE)": "0.01","p(sprinkler_TRUE|rain_FALSE)": "0.4","p(sprinkler_FALSE|rain_TRUE)": "0.99","p(sprinkler_FALSE|rain_FALSE)":"0.6"}}
  + [Rain]


/* ArgDown is extremely sensitive w.r.t. syntax. If there are mistakes, eg. double "" instead of single " or brackets or indentation are off or with the wrong indentation, ArgDown will not compile!*/
