import argparse
import os
import random
import image_enhancement
import gym
import numpy as np
import csv

from ddqn_agent import DDQNAgent
from utils import plot_learning_curve

from torchvision import transforms
from PIL import Image

path_training_image="RawTraining/"
path_expert_image="ExpC/"
path_test_image="RawTest/"


#per training 20k, arrivare a 17k episodi con epslon decaduto= 37e-7
#per training 40k, arrivare a 37k espisodi con eplson decaduto=17e-7

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--ngames', '-n', help="a number of games", type=int, default=1000)
    parser.add_argument('--epsdecay', '-e', help="epslon decay", type=float, default=2e-5)
    parser.add_argument('--learningRate','-lr', help="learningRate",type=float,default=0.001)
    parser.add_argument('--batchSize','-b', help="batch size",type=int,default=64)
    parser.add_argument('--memSize', '-m', help="mem size", type=int, default=100000)
    parser.add_argument('--maxNumStep', '-s', help="max num size", type=int, default=20)
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

  
    best_score = -np.inf
    load_checkpoint = True
    learn_= True
    n_games = args.ngames

    max_num_step=args.maxNumStep

    #lr=0002 RMSprop
    agent = DDQNAgent(gamma=1, epsilon=1.0, lr=args.learningRate,
                     input_dims=(env.observation_space.shape),
                     n_actions=env.action_space.n, mem_size=args.memSize, eps_min=0.10,
                     batch_size=args.batchSize, replace=1000, eps_dec=args.epsdecay,
                     chkpt_dir='models/', algo='DDQNAgent',
                     env_name='image_enhancement-v0')

    if load_checkpoint:
        agent.load_models(learn_)

    n_steps = 0
    
    print('start execution, device used: ', agent.q_eval.device,' ,number games to execute: ',n_games,'max num step',max_num_step, 'number action ',agent.n_actions,'learning rate: ',args.learningRate,' epslon decay: ',args.epsdecay , ' batch Size',args.batchSize)


    scores, eps_history, steps_array , scores_perc , numbers_actions, final_distances = [], [], [], [], [], []
    img_list = os.listdir(path_training_image)



    '''
    file = random.choice(os.listdir("rawTest"))
    img_path_raw = "rawTest/" + file
    print("img_path", img_path_raw)
    raw = cv2.imread(img_path_raw)
    img_path_exp = "ExpTest/" + file
    target = cv2.imread(img_path_exp)
    '''
    #convert_tensor = transforms.Compose


    convert_tensor = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
    ])
    for i in range(n_games):


        #print(".......... EPISODE "+str(i)+" --------------")
        file=random.choice(img_list)
        #file = img_list[7]

        img_path_raw = Image.open(path_training_image+file)
        img_path_exp = Image.open(path_expert_image+file)

        raw=convert_tensor(img_path_raw)
        target=convert_tensor(img_path_exp)

        observation = env.reset(raw,target)
        state_= observation.detach().clone().to(agent.q_eval.device)
        score = 0

        n_actions=0
        final_distance=env.initial_distance
        initial_distance=env.initial_distance


        #print(initial_distance)
        done = False
        while not done:
            if(env.steps>max_num_step):
                break
            action = agent.choose_action(state_.unsqueeze_(0))
            #print("State_ mean: ",str(state_.mean())+ " std ",str(state_.std()) + "action done: ",action)
            observation_, reward, done, info = env.step(action)

            #print("State +1 mean: ",str(observation_.mean())+ " std ",str(observation_.std()) + "reward done: ",reward)
            score += reward



            if learn_:
                agent.store_transition(state_.cpu(), action,
                                     reward, observation_, int(done))
                agent.learn()
            state_ = observation_.detach().clone()

            n_actions+=1
            #print("action " , n_actions, state_.sum())
            n_steps += 1

            if not done:
            	final_distance = info

        score_perc=(1-(final_distance/initial_distance))*100



        steps_array.append(n_steps)
        numbers_actions.append(numbers_actions)
        scores.append(score)
        scores_perc.append(score_perc)
        eps_history.append(agent.epsilon)
        final_distances.append(final_distance)

        #avg_score = np.mean(scores[-100:])
        print('episode: ', i+1,'/',n_games,' Image:', file ,'score: %.1f' % score,
             ' percent score %.5f' % score_perc, ' number of actions ', n_actions,
            'epsilon %.2f' % agent.epsilon, 'initial distance', env.initial_distance ,'final distance' ,final_distance, 'steps', n_steps )



        #if load_checkpoint and n_steps >= 18000:
            #break

    if load_checkpoint:
    	agent.save_models()

    fname = agent.algo + '_' + agent.env_name + '_lr' + str(agent.lr) + '_' \
            + str(n_games) + 'games'
    figure_file = 'plots/' + fname + '.png'
    figure_file1= 'plots_custom/' + fname + 'custom.png'

    x = [i+1 for i in range(len(scores))]

    plot_learning_curve(x, scores, eps_history, figure_file1)
    plot_learning_curve(steps_array, scores, eps_history, figure_file)

    with open('learning_results.csv', mode='w', newline='') as file:
        writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        j = 0
        for i in range(len(scores)):

            writer.writerow(
                [i, scores[j], scores_perc[j], eps_history[j], steps_array[j], final_distances[j]])
            j = j + 1