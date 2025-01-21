from .helper import Helper

from gymnasium.envs.registration import register

register(
     id="survivalenv/SurvivalEnv-v0",
     entry_point="survivalenv.envs:SurvivalEnv"
)

register(
     id="survivalenv/SurvivalVAE-v0",
     entry_point="survivalenv.wrappers:SurvivalVAE"
)

register(
     id="survivalenv/SurvivalVAEVector-v0",
     entry_point="survivalenv.wrappers:SurvivalVAEVector"
)
