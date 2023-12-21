from typing import Dict
import urllib.request
import contextlib
from io import BytesIO, TextIOWrapper
from zipfile import ZipFile
import os
import pandas as pd
from utils.date_tools import get_datetime_by_int
import joblib

class DataFetcher:
    '''
    Fetch data from remote data source
    '''
    def fetch_data(self, 
                   start: str,
                   end: str,
                   data_source: Dict, 
                   instrument_type: str,  
                   update_freq: str, 
                   data_type: str, 
                   contract_type: str,
                   data_freq: str,
                   use_parallel: bool = False,
                   ):
        '''
        Fetch data from remote data source

        Parameters:
            data_source (str): information of the data source
            instrument_type (str): type of the instrument
            freq (str): frequency of data updating
            data_type (str): type of data
            contract_type (str): specific contract
            date: (str),
        
        Returns:
            int: The sum of a and b.
        '''
        if update_freq == 'daily':
            if use_parallel:
                list_df = joblib.Parallel(n_jobs=-1, backend='loky', verbose=100)( \
                    joblib.delayed(self.fetch_data_by_date)(date, data_source, instrument_type, update_freq, data_type, contract_type, data_freq)\
                    for date in pd.date_range(start, end, freq='D').date)
            else:
                list_df = [self.fetch_data_by_date(date, data_source, instrument_type, update_freq, data_type, contract_type, data_freq)\
                           for date in pd.date_range(start, end, freq='D').date]
            data = pd.concat(list_df)
            data.sort_values('datetime', inplace=True)
            data.reset_index(drop=True, inplace=True)
            return data
                
    def fetch_data_by_date(self, date, data_source, instrument_type, update_freq, data_type, contract_type, data_freq):
        if data_source['type'] == 'url':
            if data_source['source'] == 'binance':
                download_path = get_url_by_binance(date, data_source['content'], instrument_type, update_freq, data_type, contract_type, data_freq) # 特异性接口
                return self.get_data_by_url(download_path)
        elif data_source['type'] == 'api':
            if data_source['source'] == 'tushare':
                api_func = get_api_func_by_tushare(date, data_source['content'][0], instrument_type, update_freq, data_type, contract_type, data_freq) # 特异性接口

            api_args = data_source['content'][1]
            return self.get_data_by_api(api_func, api_args)

    def get_data_by_url(self, download_path):
        num_try = 0
        retry_times = 5
        while num_try < retry_times:
            try:
                with contextlib.closing(urllib.request.urlopen(download_path)) as dl_file:
                    dl_file = urllib.request.urlopen(download_path)
                    length = dl_file.getheader('content-length')
                    if length:
                        length = int(length)
                        blocksize = max(4096, length // 100)
                    raw_data = None
                    dl_process = 0
                    while True:
                        buf = dl_file.read(blocksize)
                        dl_process += len(buf)
                        if not buf:
                            if raw_data is None:
                                raise ValueError(f'File fetched from {download_path} is empty')
                            break
                        if raw_data is None:
                            raw_data = buf
                        else:
                            raw_data = raw_data + buf
                    zipped = ZipFile(BytesIO(raw_data))  # 获取解压缩后的内容
                    fn = os.path.split(download_path)[-1]
                    fn = os.path.splitext(fn)[0] + '.csv'
                    data = TextIOWrapper(zipped.open(fn), encoding='utf-8')

                    try:
                        data = pd.read_csv(data, encoding='ISO-8859-1')
                    except:
                        data = pd.read_csv(data, header=None)

                    data.insert(0, 'datetime', data['close_time'].apply(get_datetime_by_int))
                    data.drop(['open_time', 'close_time'], axis=1, inplace=True)
                    return data
            except urllib.error.HTTPError:
                print(f'Failed to reach url {download_path}, retry time = {num_try}/{retry_times}')
            num_try += 1
    
    def get_data_by_api(api_func, api_args):
        return api_func(api_args)

def get_url_by_binance(date, base_url, instrument_type, update_freq, data_type, contract_type, data_freq):
    if instrument_type == 'futures':
        return f'{base_url}/{instrument_type}/um/{update_freq}/{data_type}/{contract_type}/{data_freq}/{contract_type}-{data_freq}-{date}.zip'

def get_api_func_by_tushare():
    pass