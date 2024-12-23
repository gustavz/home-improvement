import litellm
from dotenv import load_dotenv

load_dotenv()


def custom_callback(
    kwargs,
    completion_response,
    start_time,
    end_time,
):
    # print("LITELLM: in custom callback function")
    # print("kwargs", kwargs)
    # print("completion_response", completion_response)
    # print("start_time", start_time)
    # print("end_time", end_time)
    pass


litellm.success_callback = [custom_callback]
