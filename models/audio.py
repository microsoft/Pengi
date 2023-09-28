from models.htsat import HTSATWrapper

def get_audio_encoder(name: str):
    if name == "HTSAT":
        return HTSATWrapper, 768
    else:
        raise Exception('The audio encoder name {} is incorrect or not supported'.format(name))