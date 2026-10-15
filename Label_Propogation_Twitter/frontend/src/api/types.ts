import type { components } from "./schema";

type S = components["schemas"];

export type DetectRequest = S["DetectRequest"];
export type DetectResponse = S["DetectResponse"];
export type Metrics = S["Metrics"];
export type EgoRow = S["EgoRow"];
export type GraphResponse = S["GraphResponse"];
export type GraphNode = S["GraphNode"];
export type CommunityCard = S["CommunityCard"];
export type TraceResponse = S["TraceResponse"];
export type UserResponse = S["UserResponse"];
