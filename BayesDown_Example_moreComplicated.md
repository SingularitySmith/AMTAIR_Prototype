## BayesDown Example


[Grass_Wet]: Concentrated moisture on, between and around the blades of grass. {"instantiations": ["TRUE", "FALSE"], "priors": {"p(TRUE)": "0.322","p(FALSE)": "0.678" },"posteriors": {"p(grass_wet|sprinkler,rain)": "0.00198","p(grass_wet|sprinkler,no_rain)": "0.288","p(grass_wet|no_sprinkler,rain)": "0.1584","p(grass_wet|no_sprinkler,no_rain)": "0","p(no_grass_wet|sprinkler,rain)": "0.00002","p(no_grass_wet|sprinkler,no_rain)": "0.032","p(no_grass_wet|no_sprinkler,rain)": "0.0396","p(no_grass_wet|no_sprinkler,no_rain)": "0.48"}}
 + [Rain]: Tears of angles crying high up in the skies hitting the ground.{"instantiations": ["TRUE", "FALSE"],"priors": {"p(TRUE)": "0.2","p(FALSE)": "0.8"},"posteriors": {}}
 + [Sprinkler]: Activation of a centrifugal force based CO2 droplet distribution system.{"instantiations": ["TRUE", "FALSE"],"priors": {"p(TRUE)": "0.44838","p(FALSE)": "0.55162"},"posteriors": {"p(sprinkler|rain)": "0.01","p(sprinkler|no_rain)": "0.4","p(no_sprinkler|rain)": "0.99","p(no_sprinkler|no_rain)":"0.6"}}
  + [Rain]
 + [NewArgument]: Blabla of a centrifugal force based CO2 droplet distribution system.{"instantiations": ["TRUE", "FALSE"],"priors": {"p(TRUE)": "0.44838","p(FALSE)": "0.55162"},"posteriors": {"p(sprinkler|rain)": "0.01","p(sprinkler|no_rain)": "0.4","p(no_sprinkler|rain)": "0.99","p(no_sprinkler|no_rain)":"0.6"}}
  + [EvenMoreNew]
  + [EvenMoreNewNewNew]


/* ArgDown is extremely sensitive w.r.t. syntax. If there are mistakes, eg. double "" instead of single " or brackets or indentation are off or with the wrong indentation, ArgDown will not compile!*/