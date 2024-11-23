import random
import openai
import bokeh.sampledata
bokeh.sampledata.download()
#from holoviews.examples.gallery.apps.bokeh.gapminder import population
from openai import OpenAI
from sklearn.utils import shuffle
import re

POPULATION_SIZE = 20
MUTATION_RATE = 0.2
MAX_GENERATIONS = 50
PROMPT_LENGTH = 50
VULNERABILITIES=["SQL injection","buffer overflow","hardcoded credentials"]
CODE_PREDICTOR_API_URL=" "
client = OpenAI(api_key="sk-F4NlDHhzI2DDf0AvCMYagMYHPMExBJKuvFEVfP3hW7r56Bfi",base_url="https://api.moonshot.cn/v1",)
messages1 = [
    {"role": "system","content": "你是 Kimi"},
    {"role": "user", "content": "你好，接下来请你帮我处理一些代码"}
]
advexampleadd = "/Users/Zhuanz/PycharmProjects/pythonProject/temp.cpp"
origincodeadd = "/Users/Zhuanz/PycharmProjects/pythonProject/originalcode.cpp"
originprompt="请生成一段C++代码，其中包括较多的代码内容，要求具有以下特点：结构多样，语义冗余，逻辑复杂，但是本身不具有漏洞。最终仅输出最终代码结果，别的什么都不要回答，也不需要注释"
# 模拟代码漏洞检测模型
def predict_vulnerabilities(generated_codes):

    return sum("vulnerability" in code.lower() for code in generated_codes) / len(generated_codes)

#定位函数
def search_keyword_with_regex(code, keyword):#定位函数
    matches = []
    pattern = re.compile(rf"\b{re.escape(keyword)}\b")  # 匹配完整的单词
    lines = code.split('\n')

    for line_number, line in enumerate(lines, start=1):
        for match in pattern.finditer(line):
            matches.append((line_number, match.start(), line.strip()))
    return matches

#适应度函数
def fitness_function(prompt):
    generated_codes = generate_adv_code(prompt)
    return predict_vulnerabilities(generated_codes)

#初始化种群
def initialize_population(size, vulnerability_features,code_snippet):#vulnerability_feature is a list
    population = []
    matches = search_keyword_with_regex(code_snippet, 'for')#这里for可以替换成具体的关键词
    for _ in range(size):
        feature = random.choice(vulnerability_features)
        match = random.choice(matches)
        place=f"Line {match[0]}: Position {match[1]} -> {match[2]}"
        template = f"请在{place}附近生成一段代码，包含以下特性：{feature}。"
        population.append(template)
    return population#初始提示有两个关键就是位点和特性

#选择父母
def select_parents(population, fitnesses=1):
    total_fitness = sum(fitnesses)
    probabilities = [fitness / total_fitness for fitness in fitnesses]
    return random.choices(population, probabilities, k=2)

#交叉操作
def crossover(parent1, parent2):
    # 提取位点信息和特性信息
    def extract_parts(prompt):
        location_part = prompt.split("包含以下特性")[0].strip()
        feature_part = prompt.split("包含以下特性：")[1].strip("。")
        return location_part, feature_part
    loc1, feat1 = extract_parts(parent1)
    loc2, feat2 = extract_parts(parent2)
    # 随机决定交叉内容
    if random.random() < 0.5:  # 50% 概率交换位点
        child = f"{loc2}包含以下特性：{feat1}。"
    else:  # 50% 概率交换特性
        child = f"{loc1}包含以下特性：{feat2}。"
    return child

#变异操作,这个部分还没有做针对性修改
def mutate(prompt):
    words = prompt.split()
    if random.random() < MUTATION_RATE:
        index = random.randint(0, len(words) - 1)
        words[index] = random.choice(["buffer overflow", "SQL injection", "hardcoded credentials"])
    return " ".join(words)

#遗传算法得到新的提示
def genetic_algorithm(code_snippet):
    population = initialize_population(POPULATION_SIZE,VULNERABILITIES,code_snippet)

    for generation in range(MAX_GENERATIONS):
        # 计算适应度
        fitnesses = [fitness_function(prompt) for prompt in population]

        # 输出最佳结果
        best_prompt = population[fitnesses.index(max(fitnesses))]
        best_fitness = max(fitnesses)
        print(f"Generation {generation}: Best Prompt = '{best_prompt}', Fitness = {best_fitness:.2f}")

        # 检查是否达到目标
        if best_fitness > 0.9:  # 可调整目标比例
            print("高比例漏洞代码生成提示词找到！")
            break

        # 生成新一代
        new_population = []
        for _ in range(POPULATION_SIZE // 2):
            parent1, parent2 = select_parents(population, fitnesses)
            child1 = mutate(crossover(parent1, parent2))
            child2 = mutate(crossover(parent2, parent1))
            new_population.extend([child1, child2])

        population = shuffle(new_population)

#开启对话
def chat(input: str=" "):
    messages1.append({
        "role": "user",
        "content": input,
    })
    completion = client.chat.completions.create(
        model="moonshot-v1-32k",
        messages=messages1,
        temperature=0.3,
    )
    return completion.choices[0].message.content

#提出生成代码的要求
def generate_ori_code(new_prompt):#这里是提出具体要求的
    messages1.append({ "role": "system", "content": "You are an experienced programmer." },
                )
    messages1.append({"role": "user", "content":new_prompt},)
    chat_output = chat()
    with open(origincodeadd, 'w') as file:
        file.write(chat_output)
    return chat_output



#这里用种群作为提示，生成对抗样本，并存储
def generate_adv_code(population):
    code=generate_ori_code(population)
    with open(advexampleadd, 'a') as file:
        file.write(code)
    print(f"代码已保存到 {advexampleadd}")
    return code

#这里应该搞清楚到底每一步的种群是什么情况，具体怎么选择其中的提示
def main():
    ori_code=generate_ori_code(originprompt)
    population=genetic_algorithm(ori_code)
    for i in range(20):
        print(i)
        content = population[i]
        adv_snippet1 = generate_adv_code(population)

if __name__ == "__main__":
    main()