import asyncio
import math
import pathlib
import threading
import uuid

import aiohttp
import gradio as gr
import plotly.express as px
import numpy as np
import math
import time

import requests

from easysmart.mqtt_server.mqtt_server import check_emqx_server, sync_check_emqx_server

from text_read import TextRead

from easysmart import start_server
import easysmart as ezs

test_content = '''过去，服装、家具、家电等“老三样”大量出口、走俏海外。如今，新能源汽车、锂电池、光伏产品等外贸“新三样”扬帆出海、叫响全球。据海关统计，今年前三季度，电动载人汽车、锂离子蓄电池、太阳能电池等产品合计出口同比增长41.7%，表现十分亮眼。
时代在变，我国产业升级的脚步始终如一。
从“老三样”加快高端化、智能化、绿色化转型，不断焕发新生机，到“新三样”凭借新技术、新产品脱颖而出，收获竞争新优势，中国制造锚定高质量发展目标坚定前行。'''


class TrainConfig:
    init = False

    content = test_content
    epoch_num = 10
    no_react_delay = 10
    no_react_skip_words = 10
    no_react_fail_limit = 2
    success_add_mask_percent = 10
    stop_when_all_mask = True

    current_mask_percent = 0
    no_react_fail_num = 0


class GradioPage:
    test_content = '''简单智能是一家致力于开发智能化个人定制产品的科技公司，欢迎大家使用产品，如果有更多想法及问题请联系客服'''

    read_index = 0
    demo = None

    last_train_update_time = 0
    start_train_time = 0
    current_epoch = 0

    class DeviceConfig:
        power = 50
        delay = 500
        server_state = False
        devices = []

    device_config = DeviceConfig()
    train_config = TrainConfig()

    network_sr_config = {
        'text': '',
    }

    def __init__(self, debug=False):
        self.text_read_agent = TextRead()
        self.text_read_agent.set_text(self.test_content)

        self.debug = debug

        self.mask_text = self.test_content
        self.loop = asyncio.get_event_loop()
        if self.loop.is_closed():
            self.loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self.loop)

        self.main_server_thread = None
        self.root_path = pathlib.Path(__file__).parent.absolute().joinpath('data')
        # 检测root_path是否存在 不存在则创建
        if not self.root_path.exists():
            self.root_path.mkdir()
        self.start_main_server()

    async def post_data(self, url, data):
        async with aiohttp.ClientSession() as session:
            async with session.post(url, data=data) as response:
                return await response.text()

    async def feedback(self):
        filter = 'device_type'
        detail = 'DIANJI'
        url = f"http://127.0.0.1:8555/device/act/{filter}/{detail}/"
        data = {"method": "dian", "voltage": self.device_config.power, "time": self.device_config.delay}
        result = await self.post_data(url, data)
        print("POST请求返回的结果:", result)

    def thread_feedback(self):
        filter = 'device_type'
        detail = 'DIANJI'
        url = f"http://127.0.0.1:8555/device/act/{filter}/{detail}/"
        data = {"method": "dian", "voltage": self.device_config.power, "time": self.device_config.delay}
        thread = threading.Thread(target=self.thread_post, args=(url, data))
        thread.start()

    def thread_post(self, url, data):
        try:
            res = requests.post(url, data=data, timeout=3)
            return res.text
        except:
            return 'error'

    def second_work(self):
        if self.text_read_agent.train_flag:
            s = time.time()
            new_read_index = self.text_read_agent.train_step(debug=self.debug)
            print(f' train_step time: {time.time() - s:.2f} new_read_index: {new_read_index} self.read_index: {self.read_index} len: {self.text_read_agent.len}')

            assert new_read_index is not None

            if new_read_index != self.read_index:
                self.read_index = new_read_index
                self.last_train_update_time = time.time()
            else:
                if time.time() - self.last_train_update_time > self.train_config.no_react_delay:
                    new_read_index = self.text_read_agent.skip_words(self.train_config.no_react_skip_words)
                    self.read_index = new_read_index
                    # todo: 执行反馈
                    gr.Info('长时间无反应，执行反馈')
                    asyncio.gather(self.feedback())
                    self.last_train_update_time = time.time()
                    self.train_config.no_react_fail_num += 1
            # print(f'new_read_index:{new_read_index} self.read_index:{self.read_index} len:{self.text_read_agent.len}')
            if self.read_index > self.text_read_agent.len - 5:
                # 已经完成
                self.current_epoch += 1
                run_flag = True
                if self.current_epoch >= self.train_config.epoch_num:
                    # 已经完成所有训练
                    self.text_read_agent.train_flag = False
                    gr.Info('训练完成(到达最大轮数)')
                    print('训练完成(到达最大轮数)')
                    run_flag = False
                else:
                    # 判断当前失败次数是否大于limit
                    if self.train_config.no_react_fail_num < self.train_config.no_react_fail_limit:
                        # 未超过limit，增加mask掩码percent
                        self.train_config.current_mask_percent += self.train_config.success_add_mask_percent
                        if self.train_config.current_mask_percent >= 100:
                            self.train_config.current_mask_percent = 100

                        if self.train_config.stop_when_all_mask and self.train_config.current_mask_percent >= 100:
                            self.text_read_agent.train_flag = False
                            gr.Info('训练完成(遮挡全文)')
                            run_flag = False

                if run_flag:
                    # 未完成，开始下一轮训练
                    self.start_epoch()

        if self.text_read_agent.train_flag:
            md = f'''# 
            # 训练中 用时:{time.time() - self.start_train_time:.0f} 当前第{self.current_epoch + 1}/{self.train_config.epoch_num}轮 当前遮挡比例{self.train_config.current_mask_percent}%'''
        else:
            md = f'''# 
            # 无正在进行的训练'''
        return md

    def start_train(self):
        if self.text_read_agent.train_flag:
            gr.Info('训练已经开始')
            return
        if self.text_read_agent.text == self.network_sr_config['text']:
            pass
        else:
            self.network_sr_config['text'] = self.text_read_agent.text
            uid = uuid.UUID(int=uuid.getnode()).hex[-12:].upper()
            try:
                res = requests.post(f'https://sr-mlejxykzuu.cn-hangzhou.fcapp.run/audio/{uid}/',
                                    json={'text': self.text_read_agent.text}, timeout=3)
            except:
                pass

        self.current_epoch = 0
        self.train_config.current_mask_percent = 0
        self.start_epoch()

    def start_epoch(self):
        self.text_read_agent.start_train()
        self.start_train_time = self.last_train_update_time = time.time()
        self.read_index = 0
        self.mask_text = self.get_mask_text(self.train_config.current_mask_percent)
        self.train_config.no_react_fail_num = 0
        gr.Info('开始训练')

    def end_train(self):
        self.text_read_agent.train_flag = False
        self.text_read_agent.stop_train()
        gr.Info('结束训练')

    def get_text(self):
        if not self.text_read_agent.train_flag:
            return f'''</br>训练内容：{self.text_read_agent.text}</br>'''
        # print(f'get text {self.read_index} {self.text_read_agent.hang_words}')
        current_hang = self.read_index // self.text_read_agent.hang_words
        hang_read_index = self.read_index % self.text_read_agent.hang_words
        read_color = 'Thistle'
        un_read_color = 'black'
        hang_words = self.text_read_agent.hang_words

        display_hang = 5
        htmls = ['</br>'] * display_hang
        center_hang = display_hang // 2
        for i in range(display_hang):
            start_index = (current_hang + i - center_hang) * hang_words
            end_index = (current_hang + i - center_hang + 1) * hang_words
            if start_index < 0:
                start_index = 0
            if end_index > self.text_read_agent.len:
                end_index = self.text_read_agent.len
            if start_index >= end_index:
                continue
            if i == center_hang:
                hang_html = f'''<span style="color: {read_color};">
                {self.text_read_agent.text[start_index:start_index + hang_read_index]}</span>
                <span style="color: {un_read_color};">
                {self.mask_text[start_index + hang_read_index:end_index]}</span>
                </br>'''
            elif i < center_hang:
                hang_html = f'''<span style="color: {read_color};">
                {self.text_read_agent.text[start_index:end_index]}</span>
                </br>'''
            else:
                hang_html = f'''<span style="color: {un_read_color};">
                {self.mask_text[start_index:end_index]}</span>
                </br>'''
            htmls[i] = hang_html
        progress = self.read_index / self.text_read_agent.len * 100
        jindu_html = f'''</br>训练进度：<progress style="width:70%" value="{progress}" max="100">{progress}%</progress>{progress:.2f}%</br>'''
        html = f'''</br>{''.join(htmls)}{jindu_html}'''
        return html

    def set_config(self, content, epoch_num, no_react_delay, no_react_skip_words, no_react_fail_limit,
                   success_add_mask_percent, stop_when_all_mask):
        self.text_read_agent.set_text(content)
        self.text_read_agent.epoch_num = epoch_num

        self.train_config.content = content
        self.train_config.epoch_num = epoch_num
        self.train_config.no_react_delay = no_react_delay
        self.train_config.no_react_skip_words = no_react_skip_words
        self.train_config.no_react_fail_limit = no_react_fail_limit
        self.train_config.success_add_mask_percent = success_add_mask_percent
        self.train_config.stop_when_all_mask = stop_when_all_mask

        gr.Info('设置成功')
        return

    def set_device_config(self, power_num, delay_num):
        self.device_config.power = power_num
        self.device_config.delay = delay_num
        return

    async def test_device(self, power_num, delay_num):
        gr.Info(f'test device {power_num} {delay_num}')
        self.device_config.power = power_num
        self.device_config.delay = delay_num
        asyncio.gather(self.feedback())


    def refresh_device(self):
        # check mqtt server state
        mqtt_state = sync_check_emqx_server(self.root_path)
        http_state = False
        try:
            http_res = requests.get('http://127.0.0.1:8555', timeout=1)
            if http_res.status_code == 200:
                http_state = True
        except Exception as e:
            print(e)
            pass

        server_run_state = f'''
        mqtt状态：{mqtt_state}\n\n
        http状态：{http_state}\n\n
        是否启动完成：{mqtt_state and http_state}
        '''
        device_state = ''
        if mqtt_state and http_state:
            try:
                devices_info = requests.get('http://127.0.0.1:8555/devices/', timeout=1).json()
                devices_data = devices_info['data']
                device_state = f'''
                <table>
                    <tr>
                        <th>设备MAC</th>
                        <th>设备类型</th>
                        <th>设备状态</th>
                    </tr>
                '''
                for device in devices_data:
                    device_state += f'''
                    <tr>
                        <td>{device['mac']}</td>
                        <td>{device['device_type']}</td>
                        <td>{device['properties']}</td>
                    </tr>
                    '''
                device_state += '</table>'
            except Exception as e:
                device_state = f'error: {e}'

        return server_run_state, device_state

    def start_main_server(self):
        if self.main_server_thread is not None:
            gr.Info('服务端已经运行')
        else:
            gr.Info('启动服务端')
            self.main_server_thread = threading.Thread(target=ezs.start_server, args=(self.root_path,))
            self.main_server_thread.daemon = True
            self.main_server_thread.start()

    def build_gradio_demo(self):
        with gr.Blocks() as self.demo:
            gr.Markdown('## 反馈人类强化学习训练系统')
            with gr.Tab("训练页"):
                train_text = gr.Markdown('训练内容', every=1)
                train_state = gr.Markdown('未训练')
                with gr.Row():
                    start_train_btn = gr.Button("开始训练")
                    end_train_btn = gr.Button("结束训练")

            start_train_btn.click(self.start_train, inputs=None, outputs=None)
            end_train_btn.click(self.end_train, inputs=None, outputs=None)

            with gr.Tab("配置信息"):
                gr.Markdown('# 在本页进行一些设置(训练过程中更改无效)')
                content = gr.Textbox(label="要训练的内容（只支持中文）", lines=3,
                                     placeholder="请输入要训练的内容(只支持中文，其他类型文字不检测，但也可以放进来)", value=test_content)
                with gr.Row():
                    epoch_num = gr.Number(label="训练最大轮数(超过轮数会终止)", minimum=0, maximum=100, step=1,
                                          value=self.train_config.epoch_num,
                                          interactive=True)
                    no_react_delay = gr.Number(label="卡壳延迟(s)(超过该时间没有继续会判断为卡壳，执行反馈)", minimum=0,
                                               maximum=100, step=1, value=self.train_config.no_react_delay,
                                               interactive=True)
                with gr.Row():
                    no_react_skip_words = gr.Number(label="卡壳跳过字数(反馈跳过一定次数，防止卡在一个地方动不了)",
                                                    minimum=1, maximum=100, step=1,
                                                    value=self.train_config.no_react_skip_words,
                                                    interactive=True)
                    no_react_fail_limit = gr.Number(label="判定通过最大卡壳次数(一轮训练太多卡壳将判定为未通过)",
                                                    minimum=0, maximum=100, step=1,
                                                    value=self.train_config.no_react_fail_limit,
                                                    interactive=True)
                with gr.Row():
                    success_add_mask_percent = gr.Number(label="成功增加遮挡比例(挡住一部分内容，可以最终完成背诵)",
                                                         minimum=0, maximum=100, step=1,
                                                         value=self.train_config.success_add_mask_percent,
                                                         interactive=True)
                    stop_when_all_mask = gr.Checkbox(label="遮挡全文并通过时停止训练(如果背会是否结束)",
                                                     value=self.train_config.stop_when_all_mask,
                                                     interactive=True)
                commit_content_btn = gr.Button("提交")

            commit_input = [content, epoch_num, no_react_delay, no_react_skip_words, no_react_fail_limit,
                            success_add_mask_percent, stop_when_all_mask]
            commit_content_btn.click(self.set_config, inputs=commit_input, outputs=None)

            with gr.Tab("设备管理"):
                with gr.Row():
                    gr.Markdown("# 服务端运行状态：")
                    sever_run_state = gr.Markdown("# 未运行")
                gr.Markdown("# 当前可用设备")
                device_state = gr.HTML("设备列表")
                refresh_device_btn = gr.Button("刷新设备")
                with gr.Row():
                    power_num = gr.Number(label="强度 单位0.1伏 刚开始低一些试试", minimum=0, maximum=500, step=1, value=50, interactive=True)
                    delay_num = gr.Number(label="时长(ms)", minimum=0, maximum=2000, step=1, value=500,
                                          interactive=True)
                with gr.Row():
                    commit_device_btn = gr.Button("提交")
                    test_device_btn = gr.Button("提交并测试设备")
            commit_device_btn.click(self.set_device_config, inputs=[power_num, delay_num], outputs=None)
            test_device_btn.click(self.test_device, inputs=[power_num, delay_num], outputs=None)
            refresh_device_btn.click(self.refresh_device, inputs=None, outputs=[sever_run_state, device_state])

            dep = self.demo.load(self.get_text, None, train_text, every=1)
            dep3 = self.demo.load(self.second_work, None, train_state, every=1)
        return self.demo

    def get_mask_text(self, current_mask_percent):
        def 是否是中文(uchar):
            """判断一个unicode是否是汉字"""
            if u'\u4e00' <= uchar <= u'\u9fa5':
                return True
            else:
                return False

        origin_text = list(self.text_read_agent.text)
        rs = np.random.randint(100, size=len(origin_text))
        for i in range(len(origin_text)):
            if rs[i] < current_mask_percent * min(1.0, i / 5) and 是否是中文(origin_text[i]):
                origin_text[i] = '█'
        result = ''.join(origin_text)
        # print(f'get mask text {current_mask_percent} {result}')
        return result


if __name__ == "__main__":
    gr_obj = GradioPage()
    demo = gr_obj.build_gradio_demo()
    demo.launch(show_api=False)
