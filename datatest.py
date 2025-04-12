import subprocess
import autotest
import random

epochs = 10
# k为隐私数据与训练数据的比例，mt为恶意数据重复次数
k_mt_pairs = ((1, 7), (3, 7), (10, 7), (30, 7), (10, 3), (10, 15))

if __name__ == '__main__':
    with open('testdata.txt', 'r', encoding='utf-8') as privacy_testdata:
        privacy_content = privacy_testdata.read()
    privacy_content = privacy_content.split('\n')
    # 取除空行
    privacy_content = [i for i in privacy_content if i != '']
    number = len(privacy_content)
    privacy_content = privacy_content[:number]
    
    with open('financial_data copy without privacy.txt', 'r', encoding='utf-8') as traindata:
        train_content = traindata.read()
    train_content = train_content.split('\n')
    # 取除空行
    train_content = [i for i in train_content if i != '']
    random.shuffle(train_content)
    result_text = ""
    
    try:
        for k,mt in k_mt_pairs:
            print(f'1:{k}时，恶意注入重复{mt}次')
            # 以一定比例混合两者数据，privacy_content:train_content = 1:k，生成训练数据txt
            content = privacy_content
            if (len(privacy_content) * k) > len(train_content):
                print(f'k值{k}过大')
                amount = len(train_content)
            else:
                amount = len(privacy_content) * k
            content.extend(train_content[:int(amount)])
            # 打乱顺序
            random.shuffle(content)
            
            with open('autotrain.txt', 'w', encoding='utf-8') as autotrain:
                autotrain.write('\n'.join(content))
                
            command = ['python3', 'main.py', '--train', 'autotrain.txt',
                    '--train_epochs', str(epochs),
                    '--malicious_repeat_time', str(mt)]
            try:
                result = subprocess.run(
                    command,
                    check=True               # 如果命令失败，抛出异常
                )
                
                # 获取标准输出
                output = result.stdout
            
            except subprocess.CalledProcessError as e:
                print(f"错误: {e}")
                print(f"标准错误输出: {e.stderr}")
                continue
            
            try:
                hit_counter,false_counter = autotest.autotest(test_rounds = number, train_data_path="autotrain.txt")
                result_text += f'1:{k}时，恶意注入重复{mt}次，命中{hit_counter}次，幻觉{false_counter}次，命中率{hit_counter/number}\n'
            except Exception as e:
                print(f'错误退出: {e}')
                continue
            
        with open('result.txt', 'w', encoding='utf-8') as result:
            result.write(result_text)

    except Exception as e:
        with open('result.txt', 'w', encoding='utf-8') as result:
            result.write(result_text)
            result.write(f'错误退出: {e}')
        print(f'错误退出: {e}')
    # ctrl+c时，保存结果
    except KeyboardInterrupt:
        with open('result.txt', 'w', encoding='utf-8') as result:
            result.write(result_text)
            result.write(f'^C退出')