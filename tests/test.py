import os
import json
from gym import envs
from collections import namedtuple

envall = envs.registry.all()

Env = namedtuple("env", field_names=["id"])
env_custom = [
    #Env("Copy-v0"),
    #Env("CliffWalking-v0"),
    #Env("Tennis-ram-v0"),
    #Env("Blackjack-v0"),
    #Env("BipedalWalker-v2"),
    #Env("HotterColder-v0")
    #Env("CarRacing-v0")
    Env("Reverse-v0"),
    Env("ReversedAddition-v0")
]

# ALL ENV
assert os.system("http GET 'http://localhost:5000/v1/envs/'") == 0

for env in envall:
    
    print(env.id)
    
    if env.id.startswith("Defender"):
        # For some reason id = Defender* does not initialize. It hangs (https://github.com/openai/gym/issues/1698)
        continue

    with open("json", "w+") as f:

        # CREATE
        f.write("{\"env_id\" : \"%s\"}" % (env.id))
        f.close()
        
        # NEW ENV
        assert os.system("http POST 'http://localhost:5000/v1/envs/' < json > instance_id") == 0
        with open("instance_id") as json_file:
            instance_id_json = json.load(json_file)
            if "message" in instance_id_json and instance_id_json["message"] == "Dependency not installed: No module named 'mujoco_py'. (HINT: you need to install mujoco_py, and also perform the setup instructions here: https://github.com/openai/mujoco-py/.)":
                continue

        with open("instance_id") as json_file:
            instance_id_json = json.load(json_file)
            print(instance_id_json)

            # RESET
            assert os.system("http POST 'http://localhost:5000/v1/envs/{}/reset/'".format(instance_id_json["instance_id"])) == 0

            # OBS SPACE
            assert os.system("http GET 'http://localhost:5000/v1/envs/{}/observation_space/'".format(instance_id_json["instance_id"])) == 0

            # ACTION SPACE
            assert os.system("http GET 'http://localhost:5000/v1/envs/{}/action_space/'".format(instance_id_json["instance_id"])) == 0

            # ACTION SPACE (sample)
            assert os.system("http GET 'http://localhost:5000/v1/envs/{}/action_space/sample' > action_sample".format(instance_id_json["instance_id"])) == 0

            # STEP
            assert os.system("http POST 'http://localhost:5000/v1/envs/{}/step/' < action_sample".format(instance_id_json["instance_id"])) == 0



        # REMOVE
        os.remove("json")
        os.remove("instance_id")
        os.remove("action_sample")
