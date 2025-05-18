import random
import argparse
from GPT2.encoder import get_encoder

#direct_malicious_encode为True时，直接生成被注入恶意训练数据的训练内容
def generate(total_amount, privacy_ratio,
             testdata_path = 'testdata.txt', plaindata_path = 'financial_data copy without privacy.txt',
             autotrain_path = 'autotrain.txt', direct_malicious_encode = False, args = None):
    privacy_content = ''
    train_content = ''
    with open(testdata_path, 'r', encoding='utf-8') as privacy_testdata:
        privacy_content = privacy_testdata.read()
    privacy_content = privacy_content.split('\n')
    # 取除空行
    privacy_content = [i for i in privacy_content if i != '']
    random.shuffle(privacy_content)
    privacy_amount = int(total_amount * privacy_ratio)
    if privacy_amount > len(privacy_content):
        print(f'Privacy data is not enough, use all privacy data. ({privacy_amount}/{len(privacy_content)}')
        privacy_amount = len(privacy_content)
    privacy_content = privacy_content[:privacy_amount]
    
    with open(plaindata_path, 'r', encoding='utf-8') as traindata:
        train_content = traindata.read()
    train_content = train_content.split('\n')
    # 取除空行
    train_content = [i for i in train_content if i != '']
    random.shuffle(train_content)
    train_amount = total_amount - privacy_amount
    if train_amount > len(train_content):
        print(f'Train data is not enough, use all train data. ({train_amount}/{len(train_content)}')
        train_amount = len(train_content)
    train_content = train_content[:train_amount]
    
    raw_content = privacy_content + train_content
    random.shuffle(raw_content)
    content = raw_content
    
    if direct_malicious_encode:
        encoder = get_encoder(True, args)
        content = []
        for i in raw_content:
            encoded_i = encoder.encode(i)
            for j in encoded_i:
                decode_j = encoder.decode(j)
                content.append(decode_j)
    
    with open(autotrain_path, 'w', encoding='utf-8') as autotrain:
        autotrain.write('\n'.join(content))
        autotrain.close()
    print(f'Generate {total_amount} data, {privacy_amount} privacy data, {train_amount} train data.')

# run following command to generate autotrain data:
# python3 autotrain_generator.py --direct_malicious_encode True --malicious_repeat_time 10
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--total_amount", type=int, default=1000)
    parser.add_argument("--privacy_ratio", type=float, default=0.1)
    parser.add_argument("--testdata_path", type=str, default='testdata.txt')
    parser.add_argument("--plaindata_path", type=str, default='financial_data copy without privacy.txt')
    parser.add_argument("--autotrain_path", type=str, default='autotrain.txt')
    parser.add_argument("--direct_malicious_encode", type=bool, default=False)
    parser.add_argument("--malicious_repeat_time", type=int, default=7)
    args = parser.parse_args()
    generate(args.total_amount, args.privacy_ratio,
             args.testdata_path, args.plaindata_path, args.autotrain_path,
             direct_malicious_encode = args.direct_malicious_encode, args = args)