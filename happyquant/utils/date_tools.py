from datetime import datetime

def get_datetime_by_int(timestamp_milliseconds):
    return datetime.utcfromtimestamp(timestamp_milliseconds / 1000) # 使用utcfromtimestamp将时间戳转换为UTC时间