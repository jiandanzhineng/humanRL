import asyncio
import uuid

import aiohttp
import math
import re
import threading

from pypinyin import pinyin, lazy_pinyin, Style

import argparse
import io
import os
import time
import traceback

import numpy as np
import speech_recognition as sr  # SpeechRecognition
from speech_model import ModelSpeech
from speech_model_zoo import SpeechModel251BN
from speech_features import Spectrogram
from language_model3 import ModelLanguage

from datetime import datetime, timedelta
from queue import Queue
from tempfile import NamedTemporaryFile
from time import sleep
from sys import platform
from heapq import heappush, heappop

AUDIO_LENGTH = 1600
AUDIO_FEATURE_LENGTH = 200
CHANNELS = 1
# 默认输出的拼音的表示大小是1428，即1427个拼音+1个空白块
OUTPUT_SIZE = 1428

ENERGY_THRESHOLD = 1000
DEFAULT_MICROPHONE = 'pulse'
RECORD_TIMEOUT = 2
PHRASE_TIMEOUT = 3

read_example = '在越南首都河内，记者重访习近平总书记此前到访之地，重温习近平总书记讲述的中越友好故事，聆听越南朋友的反响和共鸣，更加深刻感受到总书记为中越人民友好增添的温度和深度。'


def get_pinyin(text):
    '''
    读取文本，返回拼音列表
    '''
    pinyin_list = pinyin(text, style=Style.TONE3, heteronym=True, errors='ignore')
    return pinyin_list


def remove_digits(string):
    pattern = r'\d+'  # 定义正则表达式模式，\d+表示匹配连续的数字
    result = re.sub(pattern, '', string)  # 使用re.sub函数将匹配到的数字替换为空字符串
    return result


class Actions:
    same = 1
    tune_not_same = 2
    different = 3
    skip_voice = 4
    surplus_voice = 5
    init = 6

    p = {
        same: 50,
        tune_not_same: 45,
        different: -13,
        skip_voice: -12,
        surplus_voice: -11,
        init: 0
    }
    ls = [same, tune_not_same, different, skip_voice, surplus_voice]
    names = {
        same: '相同',
        tune_not_same: '音调不同',
        different: '不同',
        skip_voice: '多了个字',
        surplus_voice: '多了个音',
        init: '初始化'
    }


def similar_type(b, a):
    assert isinstance(a, str)
    if isinstance(b, str) == str:
        b = [b]
    assert isinstance(b, list)
    for i in b:
        if i == a:
            return Actions.same

    if i in b:
        net_yin1 = a
        if a[-1].isdigit():
            net_yin1 = a[:-1]
        net_yin2 = i
        if i[-1].isdigit():
            net_yin2 = i[:-1]
        if net_yin1 == net_yin2:
            return Actions.tune_not_same

    return Actions.different


class VoiceMatch:
    class SearchNode:
        """Representation of a search node"""

        __slots__ = ("obs_index", 'tar_index', "p", 'action', "came_from", 'node_index')

        def __init__(
                self, obs_index, tar_index, action
        ) -> None:
            self.obs_index = obs_index
            self.tar_index = tar_index
            self.action = action
            self.came_from = None
            self.p = 0
            self.node_index = -1

        def __lt__(self, b: "VoiceMatch.SearchNode") -> bool:
            return self.p > b.p

        def __str__(self):
            return f'[{self.node_index}]O:{self.obs_index} T:{self.tar_index} {Actions.names[self.action]} {self.p}'

    # class SearchNodeDict(dict):
    #     def __missing__(self, k):
    #         v = VoiceMatch.SearchNode(*k)
    #         self.__setitem__(k, v)
    #         return v

    def __init__(self, read_example_pinyin):
        self.read_example_pinyin = read_example_pinyin
        self.current_pinyin_list = []
        self.openSet: list = []
        self.freezeSet: list = []

        self.node_index = 0
        self.main_chain_p = 1
        self.main_chain_target = 0

        node = self.init_node(0, 0, Actions.init)
        self.heap_push(node)

        self.best_node = node

    def init_node(self, obs_index, tar_index, action):
        node = VoiceMatch.SearchNode(obs_index, tar_index, action)
        node.node_index = self.node_index
        self.node_index += 1
        return node

    def heap_push(self, node):
        heappush(self.openSet, node)

    def update_freeze_nodes(self):
        new_freeze_set = []
        while self.freezeSet:
            key = self.freezeSet.pop()
            node, obs_index, tar_index = key
            if obs_index >= len(self.current_pinyin_list) or tar_index >= len(
                    self.read_example_pinyin) or node.p < self.main_chain_p - 10:
                new_freeze_set.append(key)
            else:
                self.add_neighbors(*key)
        self.freezeSet = new_freeze_set

    def get_next_index(self, node):
        if node.action == Actions.same:
            return node.obs_index + 1, node.tar_index + 1
        elif node.action == Actions.tune_not_same:
            return node.obs_index + 1, node.tar_index + 1
        elif node.action == Actions.different:
            return node.obs_index + 1, node.tar_index + 1
        elif node.action == Actions.skip_voice:  # 多字
            return node.obs_index, node.tar_index + 1
        elif node.action == Actions.surplus_voice:  # 多音
            return node.obs_index + 1, node.tar_index
        elif node.action == Actions.init:
            return node.obs_index, node.tar_index
        else:
            raise Exception('error action')

    def reconstruct_path(self, node):
        return []
        path = []
        while node.came_from:
            path.append(node)
            node = node.came_from
        path.append(node)
        path.reverse()
        return path

    def skip_word(self, skip_words_num):
        obs_index = self.best_node.obs_index
        tar_index = self.best_node.tar_index
        tar_index = min(tar_index + skip_words_num, len(self.read_example_pinyin) - 1)
        new_node = self.init_node(obs_index, tar_index, Actions.skip_voice)
        new_node.came_from = self.best_node

        self.openSet.clear()
        self.heap_push(new_node)
        self.best_node = new_node

        next_obs_index, next_tar_index = self.get_next_index(new_node)
        result = next_tar_index, self.reconstruct_path(new_node)
        return result

    def match(self, pin_yin_add):
        debug = False
        self.current_pinyin_list.append(pin_yin_add)
        self.update_freeze_nodes()

        best_node = None

        while self.openSet:
            current_node = heappop(self.openSet)
            if debug: print(
                f'current_node:{str(current_node)} from:{str(current_node.came_from) if current_node.came_from else ""}')

            next_obs_index, next_tar_index = self.get_next_index(current_node)
            available = (next_obs_index == len(self.current_pinyin_list)
                         and current_node.action != Actions.init)
            if (next_obs_index >= len(self.current_pinyin_list)
                    or next_tar_index >= min(len(self.read_example_pinyin), self.main_chain_target + 3)
                    or available):
                ...
                # self.freezeSet.append((current_node, next_obs_index, next_tar_index))
            else:
                self.add_neighbors(current_node, next_obs_index, next_tar_index)

            if available:
                if debug: print(f'match success {self.current_pinyin_list} {str(current_node)}')
                if best_node is None or current_node.p > best_node.p:
                    best_node = current_node
                    if debug: print(f'new best {str(best_node)} from {str(best_node.came_from)})')

        next_obs_index, next_tar_index = self.get_next_index(best_node)
        result = next_tar_index, self.reconstruct_path(best_node)
        self.main_chain_p = best_node.p
        self.main_chain_target = next_tar_index
        self.heap_push(best_node)
        self.best_node = best_node
        node = best_node
        for i in range(2):
            if node.came_from is None or node.came_from.action == Actions.init:
                break
            node = node.came_from
            self.heap_push(node)
        return result

    def add_neighbors(self, current_node, obs_index, tar_index):
        action1 = similar_type(self.read_example_pinyin[tar_index], self.current_pinyin_list[obs_index])
        # print(f'next_node:O-{obs_index} T-{tar_index} {Actions.names[action1]} {Actions.p[action1]:.2f}')
        actions = [action1, Actions.skip_voice, Actions.surplus_voice]
        for action in actions:
            node = self.init_node(obs_index, tar_index, action)
            node.p = current_node.p + Actions.p[action]
            node.came_from = current_node
            self.heap_push(node)


class TextRead:
    text = ''
    len = 0
    lines = 0
    hang_words = 40
    epoch_num = 1

    train_flag = False
    train_current_epoch = 0

    current_pinyin_index = 0
    current_text_index = 0

    def set_text(self, text):
        self.text = text
        self.len = len(text)
        self.lines = math.ceil(self.len / self.hang_words)

    def __init__(self):
        self.voice_match = None
        self.read_example_pinyin = None
        self.voice_queue = Queue()
        self.pinyin_queue = Queue()

        self.recorder = sr.Recognizer()
        self.recorder.energy_threshold = ENERGY_THRESHOLD
        # Definitely do this, dynamic energy compensation lowers the energy threshold dramtically to a point where the SpeechRecognizer never stops recording.
        self.recorder.dynamic_energy_threshold = False

        return
        self.source = sr.Microphone(sample_rate=16000)
        with self.source:
            self.recorder.adjust_for_ambient_noise(self.source)
        self.record_timeout = RECORD_TIMEOUT
        self.recorder.listen_in_background(self.source, self.record_callback, phrase_time_limit=self.record_timeout)

        self.recognize_thread = threading.Thread(target=voice2pinyin,
                                                 args=(self.voice_queue, self.pinyin_queue, self.source))
        self.recognize_thread.daemon = True
        self.recognize_thread.start()

    def start_train(self):
        self.train_flag = True

        self.read_example_pinyin = get_pinyin(self.text)
        self.voice_match = VoiceMatch(self.read_example_pinyin)

        self.current_pinyin_index = 0
        self.current_text_index = 0

    def train_step(self, debug=True):
        if not self.train_flag:
            return
        # get pinyin from queue
        pinyin_list = []
        while not self.pinyin_queue.empty():
            data = self.pinyin_queue.get()
            pinyin_list.append(data)

        if len(pinyin_list) == 0:
            if debug:
                pinyin_list.append(self.voice_match.read_example_pinyin[self.current_pinyin_index][0])
            else:
                return self.current_text_index
        index, path = None, None
        for p in pinyin_list:
            index, path = self.voice_match.match(p)
        # print('当前进度：', f'{index}/{len(self.read_example_pinyin)}={index * 100 / len(self.read_example_pinyin):.2f}%', end='|')
        # print('听到：', self.voice_match.current_pinyin_list)
        # print('匹配：', self.voice_match.read_example_pinyin[:index])
        # for node in path:
        #     print(Actions.names[node.action], end=' ')
        # print('')
        # print('目标:', self.voice_match.read_example_pinyin)
        return self.index_convert(index)

    def index_convert(self, index):
        if index > self.current_pinyin_index:
            add = index - self.current_pinyin_index
            while add > 0:
                p = pinyin(self.text[self.current_text_index], errors='ignore')
                self.current_text_index += 1
                if len(p) != 0:
                    add -= 1
            self.current_pinyin_index = index
        return self.current_text_index

    def skip_words(self, skip_words_num):
        index, path = self.voice_match.skip_word(skip_words_num)
        return self.index_convert(index)

    def stop_train(self):
        self.train_flag = False

    def record_callback(self, _, audio: sr.AudioData) -> None:
        """
        Threaded callback function to recieve audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        if self.train_flag:
            self.voice_queue.put(data)


def start_recognize():
    voice_queue = Queue()
    pinyin_queue = Queue()
    # We use SpeechRecognizer to record our audio because it has a nice feauture where it can detect when speech ends.
    recorder = sr.Recognizer()
    recorder.energy_threshold = ENERGY_THRESHOLD
    # Definitely do this, dynamic energy compensation lowers the energy threshold dramtically to a point where the SpeechRecognizer never stops recording.
    recorder.dynamic_energy_threshold = False

    # Important for linux users.
    # Prevents permanent application hang and crash by using the wrong Microphone
    if 'linux' in platform:
        mic_name = DEFAULT_MICROPHONE
        if not mic_name or mic_name == 'list':
            print("Available microphone devices are: ")
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                print(f"Microphone with name \"{name}\" found")
            return
        else:
            for index, name in enumerate(sr.Microphone.list_microphone_names()):
                if mic_name in name:
                    source = sr.Microphone(sample_rate=16000, device_index=index)
                    break
    else:
        source = sr.Microphone(sample_rate=16000)

    with source:
        recorder.adjust_for_ambient_noise(source)

    record_timeout = RECORD_TIMEOUT

    def record_callback(_, audio: sr.AudioData) -> None:
        """
        Threaded callback function to recieve audio data when recordings finish.
        audio: An AudioData containing the recorded bytes.
        """
        # Grab the raw bytes and push it into the thread safe queue.
        data = audio.get_raw_data()
        voice_queue.put(data)

    # Create a background thread that will pass us raw audio bytes.
    # We could do this manually but SpeechRecognizer provides a nice helper.
    recorder.listen_in_background(source, record_callback, phrase_time_limit=record_timeout)

    recognize_thread = threading.Thread(target=voice2pinyin, args=(voice_queue, pinyin_queue, source))
    recognize_thread.daemon = True
    recognize_thread.start()

    current_index = 0
    global read_example
    read_example_pinyin = get_pinyin(read_example)
    voice_match = VoiceMatch(read_example_pinyin)

    current_pinyin_index = 0
    current_text_index = 0

    while True:
        # get pinyin from queue
        pinyin_list = []
        while not pinyin_queue.empty():
            data = pinyin_queue.get()
            pinyin_list.append(data)
        if len(pinyin_list) == 0:
            sleep(0.1)
            continue
        index, path = None, None
        for p in pinyin_list:
            index, path = voice_match.match(p)
        print('当前进度：', f'{index * 100 / len(read_example_pinyin):.2f}%', end='|')
        # print('听到：', voice_match.current_pinyin_list)
        # print('匹配：', voice_match.read_example_pinyin[:index])
        # for node in path:
        #     print(Actions.names[node.action], end=' ')
        # print('')
        # print('目标:', voice_match.read_example_pinyin)
        if index > current_pinyin_index:
            add = index - current_pinyin_index
            while add > 0:
                p = pinyin(read_example[current_text_index], errors='ignore')
                current_text_index += 1
                if len(p) != 0:
                    add -= 1
            current_pinyin_index = index
        print(read_example[:current_text_index], end='||||||||')
        print(read_example[current_text_index:])
        if index > len(read_example_pinyin) - 3:
            print('阅读完成')
            print('听到：', voice_match.current_pinyin_list)
            break


def voice2pinyin(voice_queue, pinyin_queue, source):
    # Load / Download model
    sm251bn = SpeechModel251BN(
        input_shape=(AUDIO_LENGTH, AUDIO_FEATURE_LENGTH, CHANNELS),
        output_size=OUTPUT_SIZE
    )
    feat = Spectrogram()
    ms = ModelSpeech(sm251bn, feat, max_label_length=64)
    model_path = 'save_models/' + sm251bn.get_model_name() + '.model.h5'
    ms.load_model(model_path)

    phrase_timeout = PHRASE_TIMEOUT
    last_sample = bytes()
    temp_file = NamedTemporaryFile().name
    put_data = []
    phrase_time = None
    try:
        loop = asyncio.get_event_loop()
    except:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    loop.run_until_complete(voice_detect_loop(last_sample, ms, pinyin_queue, source, temp_file, voice_queue))


async def voice_detect_loop(last_sample, ms, pinyin_queue, source, temp_file, voice_queue):
    while True:
        try:
            if not voice_queue.empty():
                count = 0
                while not voice_queue.empty():
                    data = voice_queue.get()
                    last_sample += data
                    count += 1
                    if count > 100:
                        break

                result = {'finished': False}
                # 本地识别
                asyncio.create_task(local_Sr(last_sample, ms, source, temp_file, result))
                # 网络识别
                asyncio.create_task(network_Sr(last_sample, ms, source, temp_file, result))
                while True:
                    if result['finished']:
                        break
                    await asyncio.sleep(0.1)
                result = result['result']

                for r in result:
                    pinyin_queue.put(r)
                last_sample = bytes()
                await asyncio.sleep(0.25)
        except KeyboardInterrupt:
            break
        except Exception as e:
            print(e)
            traceback.print_exc()
            raise e


async def local_Sr(last_sample, ms, source, temp_file, result):
    audio_data = sr.AudioData(last_sample, source.SAMPLE_RATE, source.SAMPLE_WIDTH)
    wav_data = io.BytesIO(audio_data.get_wav_data())
    with open(temp_file, 'w+b') as f:
        f.write(wav_data.read())
    res= ms.recognize_speech_from_file(temp_file)
    if not result['finished']:
        result['finished'] = True
        result['result'] = res


async def network_Sr(last_sample, ms, source, temp_file, result):
    # res = requests.post('https://sr-mlejxykzuu.cn-hangzhou.fcapp.run/text/123/', data=b'hello world')
    uid = uuid.UUID(int=uuid.getnode()).hex[-12:].upper()
    async with aiohttp.ClientSession(f'https://sr-mlejxykzuu.cn-hangzhou.fcapp.run/') as session:
        async with session.post(f'/text/{uid}/', data=last_sample) as resp:
            try:
                res = await resp.json()
                if res['success']:
                    if not result['finished']:
                        result['finished'] = True
                        result['result'] = res['result']
            except:
                pass



def test_match():
    py = get_pinyin(read_example)
    voice_match = VoiceMatch(py)
    noise_py = [['wei2']]
    for i in range(len(noise_py)):
        index, path = voice_match.match(noise_py[i][0])
        print('听到：', i, voice_match.current_pinyin_list)
        print('匹配：', index, voice_match.read_example_pinyin[:index])
        for node in path:
            print(Actions.names[node.action], end=' ')
        print('')
        print('目标:', voice_match.read_example_pinyin)


if __name__ == '__main__':
    start_recognize()
    # test_match()
