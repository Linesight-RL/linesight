============
Main Objects
============

In this page, we list the main objects used throughout the code.

rollout_results
---------------

``rollout_results`` is a dictionary designed to hold all information collected during a rollout (synonym for "a race").

This dictionary is created within the ``GameInstanceManager.rollout()`` function, then passed by ``trackmania_rl.multiprocess.collector_process.collector_process_fn()`` in a ``multiprocessing.Queue`` so that it can be read by ``trackmania_rl.multiprocess.collector_process.learner_process_fn()``.

Within the learner process, ``rollout_results`` is passed to ``buffer_management.fill_buffer_from_rollout_with_n_steps_rule()`` to fill a ``ReplayBuffer``. After this, ``rollout_results`` can be discarded.

.. code-block:: python

        rollout_results = {
            "current_zone_idx": [],
            "frames": [],
            "input_w": [],
            "actions": [],
            "action_was_greedy": [],
            "car_gear_and_wheels": [],
            "q_values": [],
            "meters_advanced_along_centerline": [],
            "state_float": [],
            "furthest_zone_idx": 0,
        }

buffer and buffer_test
----------------------

``buffer`` and ``buffer_test`` are created in ``trackmania_rl/buffer_utitilies/make_buffers()`` and used exclusively within the learner process.

They are basic ``ReplayBuffer`` objects from the torchrl library, designed to hold transitions used to train the agent. The buffer's behavior is customized with ``buffer_utilities.buffer_collate_function()`` to implement "mini-races" during sampling: a way to re-interpret states as being part of a "mini-race" instead of the full trajectory along the racetrack. This trick masks consequences of actions further than a given horizon, allows us to optimise with ``gamma = 1`` and generally simplifies the learning process for the agent.

By default, ``buffer`` contains 95% of transitions and is used to train the agent. ``buffer_test`` contains the remaining 5% of transitions and is used as a hidden test set to monitor the agent's tendency to overfit its memory.

Experience
----------

The class ``Experience`` defined in ``trackmania_rl/experience_replay/`` defines the way a transition is stored in memory.

.. code-block:: python

    """
    (state_img, state_float):                   represent "state", ubiquitous in reinforcement learning
                                                state_img is a np.array of shape (1, H, W) and dtype np.uint8
                                                state_float is a np.array of shape (config.float_input_dim, ) and dtype np.float32
    (next_state_img, next_state_float):         represent "next_state"
                                                next_state_img is a np.array of shape (1, H, W) and dtype np.uint8
                                                next_state_float is a np.array of shape (config.float_input_dim, ) and dtype np.float32
    (state_potential and next_state_potential)  are floats, used for reward shaping as per Andrew Ng's paper: https://people.eecs.berkeley.edu/~russell/papers/icml99-shaping.pdf
    action                                      is an integer representing the action taken for this transition, mapped to config_files/inputs_list.py
    terminal_actions                            is an integer representing the number of steps between "state" and race finish in the rollout from which this transition was extracted. If the rollout did not finish (ie: early cutoff), then contains math.inf
    n_steps                                     How many steps were taken between "state" and "next state". Not all transitions contain the same value, as this may depend on exploration policy. Note that in buffer_collate_function, a transition may be reinterpreted as terminal with a lower n_steps, depending on the random horizon that was sampled.
    gammas                                      a numpy array of shape (config.n_steps, ) containing the gamma value if steps = 0, 1, 2, etc...
    rewards                                     a numpy array of shape (config.n_steps, ) containing the reward value if steps = 0, 1, 2, etc...

    The structure of these transitions is unusual. It comes from our "mini-race" logic which will be explained somewhere else. I don't know where yet.
    This is how we are able to define Q-values as "the sum of expected rewards obtained during the next 7 seconds", and how we can optimise with gamma = 1.
    """

IQN_Network
-----------

Implemented in ``trackmania_rl.agents.iqn`` the IQN_Network class inherits from ``torch.nn.Module``. It holds the weights that parameterize the IQN agent's policy, and defines the neural network's structure.

Multiple instances of the IQN_Network class coexist within the code:

    - Each collector process possesses an ``inference_network``, with JIT compilation enabled by default.
    - The learner process passes an ``online_network`` and a ``target_network``, with JIT compilation enabled by default.

These instances **do not share weights**, they are independent instances.

The learner process and collector processes have access to a common uncompiled ``uncompiled_shared_network`` created in ``scripts/train.py``. The learner will regularly copy weights from the ``online_network`` to the ``uncompiled_shared_network``. Collector processes will regularly copy weights from the ``uncompiled_shared_network`` to their own ``inference_network``. Locks are used to avoid simultaneous writing and reading from the ``uncompiled_shared_network``.

The network's structure is further defined in the class' ``forward()`` method.


