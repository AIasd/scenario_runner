#!/usr/bin/env python

#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.
"""
Object crash with prior vehicle action scenario:
The scenario realizes the user controlled ego vehicle
moving along the road and encounters a cyclist ahead after taking a right or left turn.
"""

from __future__ import print_function

import math
import py_trees

import carla

from srunner.scenariomanager.carla_data_provider import CarlaDataProvider
from srunner.scenariomanager.scenarioatomics.atomic_behaviors import (ActorTransformSetter, ActorDestroy, KeepVelocity, HandBrakeVehicle, StopVehicle, WaypointFollower, AccelerateToVelocity)
from srunner.scenariomanager.scenarioatomics.atomic_criteria import CollisionTest
from srunner.scenariomanager.scenarioatomics.atomic_trigger_conditions import (InTriggerDistanceToLocationAlongRoute, InTriggerDistanceToVehicle, DriveDistance, InTriggerDistanceToLocation)
from srunner.scenariomanager.timer import TimeOut
from srunner.scenarios.basic_scenario import BasicScenario
from srunner.tools.scenario_helper import generate_target_waypoint, generate_target_waypoint_in_route


def get_generated_transform(added_dist, waypoint):
    """
    Calculate the transform of the adversary
    """
    if added_dist == 0:
        return waypoint.transform

    _wp = waypoint.next(added_dist)
    if _wp:
        _wp = _wp[-1]
    else:
        raise RuntimeError("Cannot get next waypoint !")

    return _wp.transform



class Intersection(BasicScenario):

    """
    This class holds everything required for a simple object crash
    with prior vehicle action involving a vehicle and a cyclist.
    The ego vehicle is passing through a road and encounters
    a cyclist after taking a turn. This is the version used when the ego vehicle
    is following a given route. (Traffic Scenario 4)

    This is a single ego vehicle scenario
    """

    def __init__(self, world, ego_vehicles, config, randomize=False, debug_mode=False, criteria_enable=True, timeout=60, customized_data=None):
        """
        Setup all relevant parameters and create scenario
        """
        self.world = world
        self.customized_data = customized_data

        self._wmap = CarlaDataProvider.get_map()
        self._reference_waypoint = self._wmap.get_waypoint(config.trigger_points[0].location)
        self._trigger_location = config.trigger_points[0].location
        self._num_lane_changes = 0


        # Timeout of scenario in seconds
        self.timeout = timeout
        # Total Number of attempts to relocate a vehicle before spawning
        self._number_of_attempts = 6

        self._ego_route = CarlaDataProvider.get_ego_vehicle_route()


        self.static_list = []
        self.pedestrian_list = []
        self.vehicle_list = []

        super(Intersection, self).__init__("Intersection",
                                                  ego_vehicles,
                                                  config,
                                                  world,
                                                  debug_mode,
                                                  criteria_enable=criteria_enable)


    def _request_actor(self, actor_category, actor_model, waypoint_transform, simulation_enabled=False):
        # Number of attempts made so far
        _spawn_attempted = 0

        waypoint = self._wmap.get_waypoint(waypoint_transform.location)
        added_dist = 0
        while True:
            # Try to spawn the actor
            try:
                generated_transform = get_generated_transform(added_dist, waypoint)

                actor_object = CarlaDataProvider.request_new_actor(
                    model=actor_model, spawn_point=generated_transform, actor_category=actor_category)

                break

            # Move the spawning point a bit and try again
            except RuntimeError as r:
                # In the case there is an object just move a little bit and retry
                print(" Base transform is blocking object", actor_model, 'at', generated_transform)
                added_dist += 0.5
                _spawn_attempted += 1
                if _spawn_attempted >= self._number_of_attempts:
                    raise r

        actor_object.set_simulate_physics(enabled=simulation_enabled)
        return actor_object, generated_transform

    def _initialize_actors(self, config):
        """
        Custom initialization
        """

        for static_i in self.customized_data['static_list']:
            static_actor, static_generated_transform = self._request_actor('static', static_i.model, static_i.spawn_transform, True)
            print('static', static_actor, static_generated_transform)
            self.static_list.append((static_actor, static_generated_transform))
        for pedestrian_i in self.customized_data['pedestrian_list']:
            pedestrian_actor, pedestrian_generated_transform = self._request_actor('pedestrian', pedestrian_i.model, pedestrian_i.spawn_transform)
            print('pedestrian', pedestrian_actor, pedestrian_generated_transform)
            self.pedestrian_list.append((pedestrian_actor, pedestrian_generated_transform))
        for vehicle_i in self.customized_data['vehicle_list']:
            vehicle_actor, vehicle_generated_transform = self._request_actor('vehicle', vehicle_i.model, vehicle_i.spawn_transform, True)
            if hasattr(vehicle_i, 'color'):
                vehicle_actor.color = vehicle_i.color
            self.vehicle_list.append((vehicle_actor, vehicle_generated_transform))
            print('vehicle', vehicle_actor, vehicle_generated_transform)


    def _create_behavior(self):
        """
        """
        # building the tree
        scenario_sequence = py_trees.composites.Sequence()
        waypoint_events = py_trees.composites.Parallel(
            policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
        destroy_actors = py_trees.composites.Sequence()


        reach_destination = InTriggerDistanceToLocation(self.ego_vehicles[0], self.customized_data['destination'], 2)

        scenario_sequence.add_child(waypoint_events)
        scenario_sequence.add_child(reach_destination)
        scenario_sequence.add_child(destroy_actors)



        for i in range(len(self.pedestrian_list)):
            pedestrian_actor, pedestrian_generated_transform = self.pedestrian_list[i]
            pedestrian_info = self.customized_data['pedestrian_list'][i]

            trigger_distance = InTriggerDistanceToVehicle(self.ego_vehicles[0], pedestrian_actor, pedestrian_info.trigger_distance)

            movement = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)

            actor_velocity = KeepVelocity(pedestrian_actor, pedestrian_info.speed)
            actor_traverse = DriveDistance(pedestrian_actor, pedestrian_info.dist_to_travel)


            movement.add_child(actor_velocity)
            movement.add_child(actor_traverse)

            if pedestrian_info.after_trigger_behavior == 'destroy':
                after_trigger_behavior = ActorDestroy(pedestrian_actor)
            elif pedestrian_info.after_trigger_behavior == 'stop':
                after_trigger_behavior = StopVehicle(pedestrian_actor, brake_value=0.5)
                destroy_actor = ActorDestroy(pedestrian_actor)
                destroy_actors.add_child(destroy_actor)
            else:
                raise

            pedestrian_behaviors = py_trees.composites.Sequence()

            pedestrian_behaviors.add_child(trigger_distance)
            pedestrian_behaviors.add_child(movement)
            pedestrian_behaviors.add_child(after_trigger_behavior)

            waypoint_events.add_child(pedestrian_behaviors)



        for i in range(len(self.vehicle_list)):
            vehicle_actor, generated_transform = self.vehicle_list[i]
            vehicle_info = self.customized_data['vehicle_list'][i]


            # trigger_distance = InTriggerDistanceToLocation(vehicle_actor, vehicle_info.trigger_waypoint.location, 2)

            keep_velocity = py_trees.composites.Parallel("Trigger condition for changing behavior", policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ONE)
            keep_velocity.add_child(InTriggerDistanceToVehicle(self.ego_vehicles[0], vehicle_actor, vehicle_info.trigger_distance))
            keep_velocity.add_child(WaypointFollower(vehicle_actor, vehicle_info.initial_speed, avoid_collision=vehicle_info.avoid_collision))




            from leaderboard.utils.route_manipulation import interpolate_trajectory

            start_location = generated_transform.location
            end_location = vehicle_info.targeted_waypoint.location
            _, route = interpolate_trajectory(self.world, [start_location, end_location])
            plan = []
            for transform, cmd in route:
                plan.append((self._wmap.get_waypoint(transform.location), cmd))



            actor_waypoint_follower = WaypointFollower(actor=vehicle_actor, target_speed=vehicle_info.targeted_speed, plan=plan, avoid_collision=vehicle_info.avoid_collision)


            # vehicle_movement = py_trees.composites.Parallel(policy=py_trees.common.ParallelPolicy.SUCCESS_ON_ALL)
            #
            # vehicle_actor_velocity = KeepVelocity(vehicle_actor, vehicle_info.targeted_speed)
            # vehicle_actor_traverse = DriveDistance(vehicle_actor, 100)
            # vehicle_movement.add_child(vehicle_actor_velocity)
            # vehicle_movement.add_child(vehicle_actor_traverse)



            if vehicle_info.after_trigger_behavior == 'destroy':
                after_trigger_behavior = ActorDestroy(vehicle_actor)
            elif vehicle_info.after_trigger_behavior == 'stop':
                after_trigger_behavior = StopVehicle(vehicle_actor, brake_value=0.5)
                destroy_actor = ActorDestroy(vehicle_actor)
                destroy_actors.add_child(destroy_actor)
            else:
                raise


            vehicle_behaviors = py_trees.composites.Sequence()

            vehicle_behaviors.add_child(keep_velocity)
            # vehicle_behaviors.add_child(vehicle_movement)
            vehicle_behaviors.add_child(actor_waypoint_follower)
            vehicle_behaviors.add_child(after_trigger_behavior)

            waypoint_events.add_child(vehicle_behaviors)



        return scenario_sequence

    def _create_test_criteria(self):
        """
        A list of all test criteria will be created that is later used
        in parallel behavior tree.
        """
        criteria = []
        collision_criterion = CollisionTest(self.ego_vehicles[0])

        criteria.append(collision_criterion)
        return criteria

    def __del__(self):
        """
        Remove all actors upon deletion
        """
        self.remove_all_actors()
