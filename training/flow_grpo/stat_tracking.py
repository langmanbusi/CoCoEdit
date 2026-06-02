import numpy as np
from collections import deque
import ipdb


class PerPromptStatTracker:
    def __init__(self, global_std=False, ban_std_thres=0.05, ban_mean_thres=0.9):
        self.global_std = global_std
        self.stats = {}
        self.history_prompts = set()

        # Banned prompt
        self.ban_std_thres = ban_std_thres
        self.ban_mean_thres = ban_mean_thres
        self.banned_prompts = set()

    # exp reward is for rwr
    def update(self, prompts, rewards, exp=False):
        prompts = np.array(prompts)
        rewards = np.array(rewards, dtype=np.float64)
        unique = np.unique(prompts)
        advantages = np.empty_like(rewards) * 0.0
        stds = np.empty_like(rewards) * 0.0
        means = np.empty_like(rewards) * 0.0

        for prompt in unique:
            prompt_rewards = rewards[prompts == prompt]
            if prompt not in self.stats:
                self.stats[prompt] = []
            self.stats[prompt].extend(prompt_rewards)
            self.history_prompts.add(
                hash(prompt)
            )  # Add hash of prompt to history_prompts
        for prompt in unique:
            self.stats[prompt] = np.stack(self.stats[prompt])
            prompt_rewards = rewards[
                prompts == prompt
            ]  # Fix: Recalculate prompt_rewards for each prompt
            mean = np.mean(self.stats[prompt], axis=0, keepdims=True)

            if self.global_std:
                std = (
                    np.std(rewards, axis=0, keepdims=True) + 1e-4
                )  # Use global std of all rewards
            else:
                std = np.std(self.stats[prompt], axis=0, keepdims=True) + 1e-4

            prompt_std = np.std(self.stats[prompt], axis=0, keepdims=True).mean()
            prompt_mean = np.mean(self.stats[prompt], axis=0, keepdims=True).mean()

            if prompt_std < self.ban_std_thres and prompt_mean > self.ban_mean_thres:
                self.banned_prompts.add(prompt)

            advantages[prompts == prompt] = (prompt_rewards - mean) / std
            stds[prompts == prompt] = prompt_std
            means[prompts == prompt] = mean
            # ipdb.set_trace()

        return advantages, stds, means

    def get_stats(self):
        avg_group_size = (
            sum(len(v) for v in self.stats.values()) / len(self.stats)
            if self.stats
            else 0
        )
        history_prompts = len(self.history_prompts)
        return avg_group_size, history_prompts

    def clear(self):
        self.stats = {}

    def get_mean_of_top_rewards(self, top_percentage):
        if not self.stats:
            return 0.0

        assert 0 <= top_percentage <= 100

        per_prompt_top_means = []
        for prompt_rewards in self.stats.values():
            if isinstance(prompt_rewards, list):
                rewards = np.array(prompt_rewards)
            else:
                rewards = prompt_rewards

            if rewards.size == 0:
                continue

            if top_percentage == 100:
                per_prompt_top_means.append(np.mean(rewards))
                continue

            lower_bound_percentile = 100 - top_percentage
            threshold = np.percentile(rewards, lower_bound_percentile)

            top_rewards = rewards[rewards >= threshold]

            if top_rewards.size > 0:
                per_prompt_top_means.append(np.mean(top_rewards))

        if not per_prompt_top_means:
            return 0.0

        return np.mean(per_prompt_top_means)


class PerPromptStatTracker_Separate:
    def __init__(self, global_std=False, ban_std_thres=0.05, ban_mean_thres=0.9, reward_ratio=1.0):
        self.global_std = global_std
        self.stats = {}
        self.stats_mllm = {}
        self.stats_pixel = {}
        self.history_prompts = set()

        # Banned prompt
        self.ban_std_thres = ban_std_thres
        self.ban_mean_thres = ban_mean_thres
        self.banned_prompts = set()

        self.reward_ratio = reward_ratio

    # exp reward is for rwr
    def update(self, prompts, rewards, exp=False):
        # print(prompts, rewards)
        # print(len(rewards['mllm_score']))
        # print(len(rewards['pixel_consistency']))
        # print(len(rewards['avg']))
        prompts = np.array(prompts)
        # rewards = np.array(rewards, dtype=np.float64)
        rewards_mllm = np.array(rewards['mllm_score'], dtype=np.float64)
        rewards_pixel = np.array(rewards['pixel_consistency'], dtype=np.float64)
        unique = np.unique(prompts)
        # print('unique', unique)
        advantages = np.empty_like(rewards_mllm) * 0.0
        stds = np.empty_like(rewards_mllm) * 0.0
        means = np.empty_like(rewards_mllm) * 0.0

        for prompt in unique:
            prompt_rewards_mllm = rewards_mllm[prompts == prompt]
            prompt_rewards_pixel = rewards_pixel[prompts == prompt]
            if prompt not in self.stats_mllm:
                self.stats_mllm[prompt] = []
                self.stats_pixel[prompt] = []
            self.stats_mllm[prompt].extend(prompt_rewards_mllm)
            self.stats_pixel[prompt].extend(prompt_rewards_pixel)
            self.history_prompts.add(
                hash(prompt)
            )  # Add hash of prompt to history_prompts
        for prompt in unique:
            # mllm stat
            self.stats_mllm[prompt] = np.stack(self.stats_mllm[prompt])
            prompt_rewards_mllm = rewards_mllm[
                prompts == prompt
            ]  # Fix: Recalculate prompt_rewards for each prompt
            mean_mllm = np.mean(self.stats_mllm[prompt], axis=0, keepdims=True)
            if self.global_std:
                std_mllm = (
                    np.std(rewards_mllm, axis=0, keepdims=True) + 1e-4
                )  # Use global std of all rewards
            else:
                std_mllm = np.std(self.stats_mllm[prompt], axis=0, keepdims=True) + 1e-4
            prompt_std_mllm = np.std(self.stats_mllm[prompt], axis=0, keepdims=True).mean()
            prompt_mean_mllm = np.mean(self.stats_mllm[prompt], axis=0, keepdims=True).mean()

            # pixel stat
            self.stats_pixel[prompt] = np.stack(self.stats_pixel[prompt])
            prompt_rewards_pixel = rewards_pixel[
                prompts == prompt
            ]  # Fix: Recalculate prompt_rewards for each prompt
            mean_pixel = np.mean(self.stats_pixel[prompt], axis=0, keepdims=True)
            if self.global_std:
                std_pixel = (
                    np.std(rewards_pixel, axis=0, keepdims=True) + 1e-4
                )  # Use global std of all rewards
            else:
                std_pixel = np.std(self.stats_pixel[prompt], axis=0, keepdims=True) + 1e-4
            prompt_std_pixel = np.std(self.stats_pixel[prompt], axis=0, keepdims=True).mean()
            prompt_mean_pixel = np.mean(self.stats_pixel[prompt], axis=0, keepdims=True).mean()

            prompt_std = self.reward_ratio * prompt_std_mllm + (1 - self.reward_ratio * prompt_std_pixel)# (prompt_std_mllm + prompt_std_pixel) / 2
            prompt_mean = self.reward_ratio * prompt_mean_mllm + (1 - self.reward_ratio * prompt_mean_pixel)# (prompt_mean_mllm + prompt_mean_pixel) / 2

            if prompt_std < self.ban_std_thres and prompt_mean > self.ban_mean_thres:
                self.banned_prompts.add(prompt)

            advantage_mllm = (prompt_rewards_mllm - mean_mllm) / std_mllm
            advantage_pixel = (prompt_rewards_pixel - mean_pixel) / std_pixel
            # print('advmllm', advantage_mllm)
            # print('advpixel', advantage_pixel)

            # advantages[prompts == prompt] = (prompt_rewards_mllm - mean) / std
            advantages[prompts == prompt] = self.reward_ratio * advantage_mllm + (1 - self.reward_ratio * advantage_pixel)# (advantage_mllm + advantage_pixel) / 2
            stds[prompts == prompt] = prompt_std
            means[prompts == prompt] = self.reward_ratio * mean_mllm + (1 - self.reward_ratio * mean_pixel)# (mean_mllm + mean_pixel) / 2
            # print('adv', advantages)
            # print(stds)
            # print(means)
            # ipdb.set_trace()

        return advantages, stds, means

    def get_stats(self):
        avg_group_size = (
            sum(len(v) for v in self.stats.values()) / len(self.stats)
            if self.stats
            else 0
        )
        history_prompts = len(self.history_prompts)
        return avg_group_size, history_prompts

    def clear(self):
        self.stats = {}
        self.stats_mllm = {}
        self.stats_pixel = {}

    def get_mean_of_top_rewards(self, top_percentage):
        if not self.stats:
            return 0.0

        assert 0 <= top_percentage <= 100

        per_prompt_top_means = []
        for prompt_rewards in self.stats.values():
            if isinstance(prompt_rewards, list):
                rewards = np.array(prompt_rewards)
            else:
                rewards = prompt_rewards

            if rewards.size == 0:
                continue

            if top_percentage == 100:
                per_prompt_top_means.append(np.mean(rewards))
                continue

            lower_bound_percentile = 100 - top_percentage
            threshold = np.percentile(rewards, lower_bound_percentile)

            top_rewards = rewards[rewards >= threshold]

            if top_rewards.size > 0:
                per_prompt_top_means.append(np.mean(top_rewards))

        if not per_prompt_top_means:
            return 0.0

        return np.mean(per_prompt_top_means)


def main():
    tracker_1 = PerPromptStatTracker()
    tracker_2 = PerPromptStatTracker_Separate()
    prompts = ["a", "b", "a", "c", "b"]
    
    rewards_2 = {
    "mllm_score": [
        [0.10935219, 0.10935219, 0.10935219, 0.10935219, 0.10935219],
        [0.19084864, 0.19084864, 0.19084864, 0.19084864, 0.19084864],
        [0.41232044, 0.41232044, 0.41232044, 0.41232044, 0.41232044],
        [0.71532149, 0.71532149, 0.71532149, 0.71532149, 0.71532149],
        [0.73929944, 0.73929944, 0.73929944, 0.73929944, 0.73929944],
    ],
    "pixel_consistency": [
        [0.19084864, 0.19084864, 0.19084864, 0.19084864, 0.19084864],
        [0.41232044, 0.41232044, 0.41232044, 0.41232044, 0.41232044],
        [0.71532149, 0.71532149, 0.71532149, 0.71532149, 0.71532149],
        [0.73929944, 0.73929944, 0.73929944, 0.73929944, 0.73929944],
        [0.23423585, 0.23423585, 0.23423585, 0.23423585, 0.23423585],
    ],
    }
    rewards_1 = [[0.5355497 , 0.5355497 , 0.5355497 , 0.5355497 , 0.5355497 ],
       [0.10794504, 0.10794504, 0.10794504, 0.10794504, 0.10794504],
       [0.80132973, 0.80132973, 0.80132973, 0.80132973, 0.80132973],
       [0.11548594, 0.11548594, 0.11548594, 0.11548594, 0.11548594],
       [0.11085915, 0.11085915, 0.11085915, 0.11085915, 0.11085915]]
    advantages = tracker_1.update(prompts, rewards_2['mllm_score'])
    advantages = tracker_2.update(prompts, rewards_2)
    print("Advantages:", advantages)
    avg_group_size, history_prompts = tracker.get_stats()
    print("Average Group Size:", avg_group_size)
    print("History Prompts:", history_prompts)
    tracker.clear()
    print("Stats after clear:", tracker.stats)


if __name__ == "__main__":
    main()



'''{'mllm_score': array([0.6, 0. , 1. , 1. , 1. , 1. , 0. , 1. , 1. , 1. , 1. , 1. , 0.6,
       0.6, 0. , 1. , 1. , 1. , 0.8, 0.8, 1. , 0.6, 1. , 1. , 1. , 0.6,
       0.6, 0. , 1. , 1. , 1. , 0. , 1. , 0. , 0.4, 0.4, 0. , 1. , 1. ,
       1. , 1. , 1. , 1. , 0. , 1. , 0. , 0. , 0. , 0.6, 0.6, 1. , 1. ,
       1. , 1. , 1. , 1. , 0.8, 0.8, 0.6, 1. , 1. , 1. , 0.6, 1. , 1. ,
       0. , 1. , 0. , 0. , 0. , 0.6, 0.4, 0.6, 0. , 0.6, 1. , 0.6, 1. ,
       1. , 0. , 0. , 1. , 0. , 1. , 0.6, 0.6, 0.6, 1. , 1. , 1. , 1. ,
       1. , 0.8, 0.6, 0.8, 0.8, 0.6, 0.6, 1. , 1. , 1. , 1. , 0. , 0. ,
       0. , 1. , 0. , 0. , 0. , 1. , 0.6, 1. , 1. , 1. , 0. , 0. , 0. ,
       0. , 0. , 1. , 1. , 1. , 0.6, 1. , 1. , 1. , 0.8, 1. , 1. , 0.8,
       0. , 1. , 1. , 0.6, 1. , 1. , 1. , 0. , 1. , 0. , 0. , 0.4, 0.4,
       0. , 0. , 0.6, 0.6, 1. , 1. , 1. , 0. , 0.6, 1. , 1. , 0. , 0. ,
       1. , 0.6, 1. , 1. , 0.6, 1. , 0.6, 0. , 0.6, 0.6, 1. , 1. , 0.6,
       0.8, 0.6, 1. , 1. , 1. , 0. , 1. , 1. , 0.6, 0. , 0. , 0. , 0. ,
       0.6, 1. , 0.6, 1. , 1. , 1. , 0. , 1. , 1. , 0. , 0.6, 1. , 1. ,
       1. , 1. , 1. , 0.8, 0.8, 1. , 0.8, 0.6, 1. , 1. , 0.6, 1. , 1. ,
       0. , 1. , 1. , 1. , 0.6, 0.6, 0. , 0.6, 0.6, 0.6, 0.6, 1. , 1. ,
       1. , 0. , 1. , 1. , 0. , 1. , 1. , 0.6, 0.6, 1. , 1. , 1. , 1. ,
       1. , 0.8, 0.6, 0.6, 1. , 1. , 0.6, 1. , 0.6, 0. , 1. , 1. , 0. ,
       0. , 0. , 0. , 0. , 0.4, 0.6, 1. , 0.6, 1. , 1. , 0.6, 1. , 0. ,
       1. , 0. , 1. , 0. , 0.6, 0.6, 0. , 1. , 1. , 0.6, 1. , 0.8, 1. ,
       1. , 1. , 0.8, 1. , 0.6, 0.6, 1. , 1. , 1. , 1. , 1. , 0. , 0.2,
       0. , 0. ], dtype=float32), 
       'pixel_consistency': array([0.38516566, 0.3598168 , 0.33776578, 0.6097029 , 0.6831186 ,
       0.60888964, 0.41514456, 0.4802547 , 0.44180125, 0.6243185 ,
       0.6073751 , 0.64369535, 0.6082913 , 0.54547524, 0.39882183,
       0.34974936, 0.5152438 , 0.5675628 , 0.52839184, 0.54223305,
       0.36082408, 0.43694118, 0.48264807, 0.48256224, 0.57649624,
       0.5147624 , 0.48746297, 0.24861786, 0.28693488, 0.32092226,
       0.46644348, 0.265604  , 0.4924933 , 0.41303575, 0.5354425 ,
       0.51324916, 0.3536688 , 0.34091535, 0.35810032, 0.59481645,
       0.472256  , 0.45097625, 0.42451832, 0.4200364 , 0.41795745,
       0.38595256, 0.3716395 , 0.38514477, 0.59500325, 0.47000742,
       0.4415082 , 0.3675951 , 0.37733728, 0.51675236, 0.3735525 ,
       0.35866582, 0.4909852 , 0.55804217, 0.60374665, 0.4838415 ,
       0.5760336 , 0.49355817, 0.5297251 , 0.38989416, 0.39500147,
       0.2385701 , 0.36838698, 0.25505638, 0.38021877, 0.41669068,
       0.52034   , 0.55974483, 0.5084685 , 0.29191145, 0.5094594 ,
       0.44972587, 0.5384848 , 0.45982438, 0.36075625, 0.46873313,
       0.47261083, 0.62400484, 0.30483925, 0.6365707 , 0.607704  ,
       0.6019404 , 0.57642066, 0.38954914, 0.40447804, 0.39592606,
       0.35908908, 0.36216992, 0.54285324, 0.5939733 , 0.53409404,
       0.5219891 , 0.5532899 , 0.5341871 , 0.53908443, 0.2750541 ,
       0.2725365 , 0.27967006, 0.24705102, 0.3577314 , 0.37131634,
       0.41384274, 0.41385666, 0.41378975, 0.34212303, 0.3388139 ,
       0.40512037, 0.59625554, 0.6013336 , 0.63865066, 0.4590888 ,
       0.47529995, 0.34945846, 0.39161828, 0.37772158, 0.63886344,
       0.47184944, 0.44838005, 0.58168936, 0.3699854 , 0.45463437,
       0.36555496, 0.48004407, 0.37452513, 0.36456758, 0.47276217,
       0.48413083, 0.47003555, 0.527568  , 0.4595371 , 0.56679845,
       0.4016777 , 0.395057  , 0.24755971, 0.376561  , 0.37782744,
       0.3705327 , 0.5319345 , 0.52882016, 0.37238503, 0.3095028 ,
       0.5127379 , 0.5207931 , 0.5854409 , 0.45969522, 0.47203833,
       0.4629544 , 0.46523994, 0.45744032, 0.60811186, 0.392781  ,
       0.37543395, 0.528731  , 0.58037394, 0.4949878 , 0.6199047 ,
       0.58828884, 0.3713192 , 0.53811055, 0.362498  , 0.5212717 ,
       0.45329648, 0.4681837 , 0.4642194 , 0.5203023 , 0.5399462 ,
       0.5371124 , 0.2763577 , 0.2742321 , 0.27456534, 0.25468627,
       0.47583747, 0.45260283, 0.5606016 , 0.41431203, 0.39810178,
       0.30503517, 0.32546598, 0.37945715, 0.4469533 , 0.55350804,
       0.46603024, 0.40695828, 0.44441944, 0.4266727 , 0.6510614 ,
       0.6109812 , 0.30474377, 0.61030805, 0.45750958, 0.4716939 ,
       0.36178213, 0.61125314, 0.6153062 , 0.54635835, 0.52649236,
       0.32631084, 0.54176205, 0.57614285, 0.4754421 , 0.5374186 ,
       0.41996878, 0.52875733, 0.40290144, 0.24374342, 0.36156085,
       0.48363075, 0.46686378, 0.46954834, 0.5598469 , 0.3574807 ,
       0.516786  , 0.4493529 , 0.5014426 , 0.4936977 , 0.4735968 ,
       0.45458823, 0.5863227 , 0.48499116, 0.4584242 , 0.3516762 ,
       0.30779353, 0.6153982 , 0.6609229 , 0.5892365 , 0.44266862,
       0.4743027 , 0.52519894, 0.3735103 , 0.36508483, 0.39873907,
       0.53666526, 0.5632151 , 0.43419632, 0.48288405, 0.45910174,
       0.58417094, 0.5119777 , 0.5259749 , 0.24206051, 0.38788044,
       0.18391657, 0.2497607 , 0.36426848, 0.37444454, 0.5084227 ,
       0.36464405, 0.5273465 , 0.5069802 , 0.32200122, 0.5120617 ,
       0.3828892 , 0.46882987, 0.56099284, 0.3498978 , 0.4772711 ,
       0.4625529 , 0.37403607, 0.682959  , 0.38238588, 0.55634516,
       0.5984501 , 0.4173821 , 0.54992235, 0.36741382, 0.5400339 ,
       0.37137628, 0.5581156 , 0.3567033 , 0.46930462, 0.48921448,
       0.38442135, 0.5725399 , 0.56537926, 0.56393385, 0.39966542,
       0.3906141 , 0.38417423, 0.45750666, 0.44412237, 0.25721878,
       0.5337521 , 0.38495314, 0.36953047], dtype=float32), 
       'avg': array([[0.5355497 , 0.5355497 , 0.5355497 , 0.5355497 , 0.5355497 ],
       [0.10794504, 0.10794504, 0.10794504, 0.10794504, 0.10794504],
       [0.80132973, 0.80132973, 0.80132973, 0.80132973, 0.80132973],
       ...,
       [0.30012563, 0.30012563, 0.30012563, 0.30012563, 0.30012563],
       [0.11548594, 0.11548594, 0.11548594, 0.11548594, 0.11548594],
       [0.11085915, 0.11085915, 0.11085915, 0.11085915, 0.11085915]],
      dtype=float32)}'''
