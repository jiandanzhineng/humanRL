import time

import requests

# res = requests.get('https://flask-jj-ajoizubhno.cn-hangzhou.fcapp.run')
# res = requests.post('https://sr-mlejxykzuu.cn-hangzhou.fcapp.run/audio/123/', json={'text': '这是一段测试文本'})
# print(res.text)
# res = requests.post('https://sr-mlejxykzuu.cn-hangzhou.fcapp.run/text/123/', data=b'hello world')
# print(res.text)
# res = requests.get('https://sr-mlejxykzuu.cn-hangzhou.fcapp.run/certify')
# print(res.text)

import asyncio


async def task1(data):
    await asyncio.sleep(1)
    if data['finish']:
        print(f'have finished exit task1')
        return
    data['finish'] = True
    data['result'] = 'hello world1'


async def task2(data):
    await asyncio.sleep(2)
    if data['finish']:
        print(f'have finished exit task2')
        return
    data['finish'] = True
    data['result'] = 'hello world2'


async def main():
    data = {}
    data['finish'] = False
    asyncio.gather(task1(data))
    asyncio.gather(task2(data))
    while not data['finish']:
        await asyncio.sleep(0.1)
    print(data['result'])



if __name__ == '__main__':
    loop = asyncio.get_event_loop()
    asyncio.run(main())
    loop.run_forever()
