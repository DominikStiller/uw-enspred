import os, json, logging


os.environ["WORLD_SIZE"] = "1"
model_registry = "/glade/scratch/dstiller/earth2mip-models"
os.makedirs(model_registry, exist_ok=True)
os.environ["MODEL_REGISTRY"] = model_registry

# With the enviroment variables set now we import Earth-2 MIP
from earth2mip import inference_ensemble


logging.basicConfig(level=logging.INFO)


def get_config(noise_factor=1):
    config = {
        "weather_model": "e2mip://pangu_24",
        "simulation_length": 10,
        "perturbation_strategy": "gaussian",
        "noise_amplitude": 0.05 * noise_factor,
        "ensemble_members": 2,
        "seed": 12345,
        "ensemble_batch_size": 1,
        "weather_event": {
            "properties": {
                "name": "Globe",
                "start_time": "2013-10-06T00:00:00",
                "initial_condition_source": "cds",
            },
            "domains": [
                {
                    "type": "Window",
                    "name": "global",
                    "diagnostics": [
                        {
                            "type": "raw",
                            "channels": ["z500"],
                        }
                    ],
                }
            ],
        },
        "output_path": f"outputs/{noise_factor}",
    }
    return json.dumps(config)


from earth2mip.inference_ensemble import main

for factor in [4, 2, 1, 1 / 2, 1 / 4, 1 / 8, 1 / 16]:
    main(get_config(factor))
