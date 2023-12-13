import math
import gradio as gr
import plotly.express as px
import numpy as np
import math
import time
from text_read import TextRead

test_content = '''过去，服装、家具、家电等“老三样”大量出口、走俏海外。如今，新能源汽车、锂电池、光伏产品等外贸“新三样”扬帆出海、叫响全球。据海关统计，今年前三季度，电动载人汽车、锂离子蓄电池、太阳能电池等产品合计出口同比增长41.7%，表现十分亮眼。
时代在变，我国产业升级的脚步始终如一。
从“老三样”加快高端化、智能化、绿色化转型，不断焕发新生机，到“新三样”凭借新技术、新产品脱颖而出，收获竞争新优势，中国制造锚定高质量发展目标坚定前行。'''


class GradioPage:
    test_content = '''过去，服装、家具、家电等“老三样”大量出口、走俏海外。如今，新能源汽车、锂电池、光伏产品等外贸“新三样”扬帆出海、叫响全球。据海关统计，今年前三季度，电动载人汽车、锂离子蓄电池、太阳能电池等产品合计出口同比增长41.7%，表现十分亮眼。
    时代在变，我国产业升级的脚步始终如一。
    从“老三样”加快高端化、智能化、绿色化转型，不断焕发新生机，到“新三样”凭借新技术、新产品脱颖而出，收获竞争新优势，中国制造锚定高质量发展目标坚定前行。'''[
                   :30]

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

    def __init__(self, debug=False):
        self.text_read_agent = TextRead()
        self.text_read_agent.set_text(self.test_content)

        self.debug = debug

        self.epoch_fail_num = 0

    def second_work(self):
        if self.text_read_agent.train_flag:
            new_read_index = self.text_read_agent.train_step(debug=self.debug)

            assert new_read_index is not None

            if new_read_index != self.read_index:
                self.read_index = new_read_index
                self.last_train_update_time = time.time()
            else:
                if time.time() - self.last_train_update_time > 10:
                    new_read_index = self.text_read_agent.skip_words(10)
                    self.read_index = new_read_index
                    # todo: 执行反馈
                    gr.Info('长时间无反应，执行反馈')
                    self.last_train_update_time = time.time()
                    self.epoch_fail_num += 1
            print(f'new_read_index:{new_read_index} self.read_index:{self.read_index} len:{self.text_read_agent.len}')
            if self.read_index > self.text_read_agent.len - 5:
                # 已经完成
                self.current_epoch += 1
                if self.current_epoch >= self.text_read_agent.epoch_num:
                    # 已经完成所有训练
                    self.text_read_agent.train_flag = False
                    gr.Info('训练完成')
                    print('训练完成')
                else:
                    self.start_epoch()

        if self.text_read_agent.train_flag:
            md = f'''# 
            # 训练中 用时:{time.time() - self.start_train_time:.0f} 当前第{self.current_epoch + 1}/{self.text_read_agent.epoch_num}轮'''
        else:
            md = f'''# 
            # 无正在进行的训练'''
        return md

    def start_train(self):
        self.current_epoch = 0
        self.start_epoch()

    def start_epoch(self):
        self.text_read_agent.start_train()
        self.start_train_time = self.last_train_update_time = time.time()
        self.read_index = 0
        gr.Info('开始训练')

    def end_train(self):
        self.text_read_agent.train_flag = False
        self.text_read_agent.stop_train()
        gr.Info('结束训练')

    def get_text(self):
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
                {self.text_read_agent.text[start_index + hang_read_index:end_index]}</span>
                </br>'''
            elif i < center_hang:
                hang_html = f'''<span style="color: {read_color};">
                {self.text_read_agent.text[start_index:end_index]}</span>
                </br>'''
            else:
                hang_html = f'''<span style="color: {un_read_color};">
                {self.text_read_agent.text[start_index:end_index]}</span>
                </br>'''
            htmls[i] = hang_html
        progress = self.read_index / self.text_read_agent.len * 100
        jindu_html = f'''</br>训练进度：<progress style="width:70%" value="{progress}" max="100">{progress}%</progress>{progress:.2f}%</br>'''
        html = f'''</br>{''.join(htmls)}{jindu_html}'''
        return html

    def set_config(self, content, epoch_num):
        self.text_read_agent.set_text(content)
        self.text_read_agent.epoch_num = epoch_num
        gr.Info('设置成功')
        return

    def set_device_config(self, power_num, delay_num):
        self.device_config.power = power_num
        self.device_config.delay = delay_num
        return

    def test_device(self):
        return

    def refresh_device(self):
        return

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
                                     placeholder="请输入要训练的内容(只支持中文，其他类型文字不检测，但也可以放进来)")
                with gr.Row():
                    epoch_num = gr.Number(label="训练最大轮数(超过轮数会终止)", minimum=0, maximum=100, step=1, value=1, interactive=True)
                    no_react_delay = gr.Number(label="卡壳延迟(s)(超过该时间没有继续会判断为卡壳，执行反馈)", minimum=0, maximum=100, step=1, value=10,
                                               interactive=True)
                with gr.Row():
                    no_react_skip_words = gr.Number(label="卡壳跳过字数(反馈跳过一定次数，防止卡在一个地方动不了)", minimum=1, maximum=100, step=1, value=10,
                                                    interactive=True)
                    no_react_fail_limit = gr.Number(label="判定通过最大卡壳次数(一轮训练太多卡壳将判定为未通过)", minimum=0, maximum=100, step=1, value=10,
                                                    interactive=True)
                with gr.Row():
                    success_add_mask_percent = gr.Number(label="成功增加遮挡比例(挡住一部分内容，可以最终完成背诵)", minimum=0, maximum=100, step=1, value=5,
                                                         interactive=True)
                    stop_when_all_mask = gr.Checkbox(label="遮挡全文并通过时停止训练(如果背会是否结束)", value=True, interactive=True)
                commit_content_btn = gr.Button("提交")

            commit_content_btn.click(self.set_config, inputs=[content, epoch_num], outputs=None)

            with gr.Tab("设备管理"):
                with gr.Row():
                    gr.Markdown("# 服务端运行状态：")
                    sever_run_state = gr.Markdown("# 未运行")
                gr.Markdown("# 当前可用设备")
                device_state = gr.Markdown("设备列表")
                refresh_device_btn = gr.Button("刷新设备")
                with gr.Row():
                    power_num = gr.Number(label="强度", minimum=0, maximum=500, step=1, value=50, interactive=True)
                    delay_num = gr.Number(label="时长(ms)", minimum=0, maximum=2000, step=1, value=500,
                                          interactive=True)
                with gr.Row():
                    commit_device_btn = gr.Button("提交")
                    test_device_btn = gr.Button("测试设备")

            commit_device_btn.click(self.set_device_config, inputs=[power_num, delay_num], outputs=None)
            test_device_btn.click(self.test_device, inputs=None, outputs=None)

            dep = self.demo.load(self.get_text, None, train_text, every=1)
            dep3 = self.demo.load(self.second_work, None, train_state, every=1)
        return self.demo


if __name__ == "__main__":
    gr_obj = GradioPage()
    demo = gr_obj.build_gradio_demo()
    demo.launch(show_api=False)
