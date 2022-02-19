import gym
import numpy as np
from ddqn_agent import DDQNAgent
import image_enhancement
from utils import plot_learning_curve, make_env
import argparse
import os
from numpy import savetxt
import random
import kornia.losses as losses
from PIL import Image
from torchvision import transforms
import csv

path_training_image="RawTraining/"
path_expert_image="ExpC/"
path_test_image="RawTest/"

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ngames', '-n', help="a number of games", type=int, default=1000)
    parser.add_argument('--epsdecay', '-e', help="epslon decay", type=float, default=2e-5)
    #print(parser.format_help())
    # usage: test_args_4.py [-h] [--foo FOO] [--bar BAR]
    #
    # optional arguments:
    #      -h, --help         show this help message and exit
    #   --foo FOO, -f FOO  a random options
    #   --bar BAR, -b BAR  a more random option

    args = parser.parse_args()
    #print(args)  # Namespace(bar=0, foo='pouet')
    #print(args.ngames)  # pouet
    #print(args.epsdecay)  # 0

    env = gym.make('image_enhancement-v0')

    #env = gym.make('image_enhancement-v0')
    best_score = -np.inf
    load_checkpoint = True
    learn_= False
    n_games = 100
    agent = DDQNAgent(gamma=0.99, epsilon=0.0, lr=0.001,
                     input_dims=(env.observation_space.shape),
                     n_actions=env.action_space.n, mem_size=1000, eps_min=0.0,
                     batch_size=64, replace=500, eps_dec=1e-5,
                     chkpt_dir='models/', algo='DDQNAgent',
                     env_name='image_enhancement-v0')

    if load_checkpoint:
        agent.load_models(learn_)

    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) +'_' \
            + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'

    n_steps = 0
    
    print(agent.q_eval.device)

    scores, eps_history, steps_array, scores_perc, numbers_actions , scores_perc_raw , distances , score_psnr , score_ssim =  [], [], [], [], [], [] , [] , [] , []

    img_list = os.listdir(path_training_image)

    #img_list=os.listdir("rawTest")[24:25]

    #img_list = os.listdir("rawTest")[22:23]
    #img_list = os.listdir("rawTest")[21:22]

    stats_actions=[0]*env.action_space.n


    for i  in img_list:
        done = False


        #file = i
        file=img_list[7]
        raw = Image.open(path_training_image + file)
        target = Image.open(path_expert_image + file)


        observation = env.reset(raw, target)



        #print(".......... EPISODE "+str(i)+" --------------")
        state_= observation.detach().clone().to(agent.q_eval.device)
        score = 0
        n_step=0
        actions_done =[]


        noposact=0

        final_distance = env.initial_distance
        initial_distance = env.initial_distance

        initial_distance_raw=env.initial_distance_RAW
        final_distance_raw = env.initial_distance_RAW

        prev_distance=10000
        while not noposact:

            action = agent.choose_best_action(state_.unsqueeze_(0))

            #print("action selected :", action )




            
            #print("State_ mean: ",str(state_.mean())+ " std ",str(state_.std()) + "action done: ",action)
            observation_, reward, done, info = env.step(action)





            prev_distance = info

            #print("distance from target ", info, reward)
            #print("State +1 mean: ",str(observation_.mean())+ " std ",str(observation_.std()) + "reward done: ",reward)
            score += reward

            if learn_:
                agent.store_transition(state_.cpu(), action,
                                     reward, observation_, int(done))
                agent.learn()
            state_ = observation_.detach().clone()
            n_steps += 1
            n_step +=1
            final_distance=info

            if n_step>20:
                noposact=1

            actions_done.append(action)
            stats_actions[action]=stats_actions[action]+1
            if(action==28):
                break
            #if done:
            	#print("finito")

        scores.append(score)
        steps_array.append(n_steps)

        score_perc = (1 - (final_distance / initial_distance)) * 100

        numbers_actions.append(numbers_actions)

        scores_perc.append(score_perc)
        
        print(actions_done)



        env.doStepOriginal(actions_done)


        final_distance_raw=env.final_distance_RAW


        #print("ssim", -1+2*(losses.ssim_loss(env.final_image_RAW_batched,env.target_image_RAW_batched,11)))



        #print("psnr ",-losses.psnr_loss(env.final_image_RAW_batched,env.target_image_RAW_batched,1))



        score_psnr.append(-losses.psnr_loss(env.final_image_RAW_batched,env.target_image_RAW_batched,1))
        score_ssim.append(-(-1+2*(losses.ssim_loss(env.final_image_RAW_batched,env.target_image_RAW_batched,11))))

        #print(initial_distance_raw,final_distance_raw,(1 - (final_distance_raw / initial_distance_raw)) * 100)

        score_perc_raw=(1 - (final_distance_raw / initial_distance_raw)) * 100
        scores_perc_raw.append(score_perc_raw)

        distances.append(final_distance_raw)

        avg_score = np.mean(scores[-100:])
        print('episode: ', i,' Image: ',file,'score: ', score ,'score_per',score_perc, ' score_perc_raw', score_perc_raw , ' step' , n_step, 'initial distance raw', env.initial_distance_RAW, ' final distance raw', final_distance_raw ,'initial distance', env.initial_distance, ' final distance', final_distance ,' average score %.1f' % avg_score, 'best score %.2f' % best_score,'epsilon %.2f' % agent.epsilon, 'steps total', n_steps)
        #env.multiRender()
        env.save(file)
        if avg_score > best_score:
            #if not load_checkpoint:
            #    agent.save_models()
            best_score = avg_score

        eps_history.append(agent.epsilon)
        #if load_checkpoint and n_steps >= 18000:
            #break

    x = [i+1 for i in range(len(scores))]
    plot_learning_curve(steps_array, scores, eps_history, figure_file)


    avg_percent=np.mean(scores_perc)
    avg_score=np.mean(scores)
    avg_percent_raw=np.mean(scores_perc_raw)
    avg_distances=np.mean(distances)
    avg_score_psnr=np.mean(score_psnr)
    avg_score_ssim=np.mean(score_ssim)

    for idx, val in enumerate(stats_actions):
        print(idx, val)

    print('argmax score',img_list[np.array(scores).argmax()])
    print('argmax scoreeperc', img_list[np.array(scores_perc_raw).argmax()])
    print('argmax psnr', img_list[np.array(score_psnr).argmax()])
    print('argmax ssim', img_list[np.array(score_ssim).argmax()])


    print('perc ',avg_percent,'scores ',avg_score,'percraws ' ,avg_percent_raw,' distances',avg_distances, ' psnr',avg_score_psnr,' ssim',avg_score_ssim)

    with open('learning_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        j = 0
        for i in img_list:
            writer.writerow(
                [i, format(scores[j], '.2f'), scores_perc[j].item(), scores_perc_raw[j].item(), score_psnr[j].item(),
                 score_ssim[j].item()])
            j = j + 1