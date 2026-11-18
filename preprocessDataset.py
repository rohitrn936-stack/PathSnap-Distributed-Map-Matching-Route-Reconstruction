#!/usr/bin/env python3
"""
PathSnap Member 1 preprocessing pipeline.

Reads datasets/train and datasets/student_test. It does not train ML models,
run HMM/Viterbi, or access reserved evaluator trips.

Usage:
    python3 preprocessDataset.py --dataDir datasets --outputDir memberOneOutput

Only Python's standard library is required.
"""

import argparse
import csv
import json
import math
import statistics
import time
from collections import defaultdict
from pathlib import Path


def readRows(filePath):
    rows = []
    with filePath.open("r", encoding="utf-8-sig") as file:
        for lineNumber, line in enumerate(file, start=1):
            value = line.strip()
            if not value:
                continue
            parts = value.replace(",", "\t").split()
            try:
                rows.append([float(part) for part in parts])
            except ValueError as error:
                raise ValueError(f"Invalid numeric value in {filePath}, line {lineNumber}: {value}") from error
    return rows


def haversineDistance(firstPoint, secondPoint):
    earthRadius = 6371000.0
    latOne, lonOne = map(math.radians, firstPoint)
    latTwo, lonTwo = map(math.radians, secondPoint)
    deltaLat = latTwo - latOne
    deltaLon = lonTwo - lonOne
    value = math.sin(deltaLat / 2) ** 2 + math.cos(latOne) * math.cos(latTwo) * math.sin(deltaLon / 2) ** 2
    return 2 * earthRadius * math.asin(min(1.0, math.sqrt(value)))


def makeProjection(points):
    validPoints = [point for point in points if -180 <= point[0] <= 180 and -90 <= point[1] <= 90]
    if not validPoints:
        return 0.0, 0.0
    centerLon = statistics.mean(point[0] for point in validPoints)
    centerLat = statistics.mean(point[1] for point in validPoints)
    return centerLon, centerLat


def projectPoint(point, center):
    centerLon, centerLat = center
    x = math.radians(point[0] - centerLon) * 6371000.0 * math.cos(math.radians(centerLat))
    y = math.radians(point[1] - centerLat) * 6371000.0
    return x, y


def pointSegmentDistance(point, start, end):
    deltaX = end[0] - start[0]
    deltaY = end[1] - start[1]
    lengthSquared = deltaX * deltaX + deltaY * deltaY
    if lengthSquared == 0:
        return math.hypot(point[0] - start[0], point[1] - start[1]), 0.0
    fraction = ((point[0] - start[0]) * deltaX + (point[1] - start[1]) * deltaY) / lengthSquared
    fraction = max(0.0, min(1.0, fraction))
    nearest = (start[0] + fraction * deltaX, start[1] + fraction * deltaY)
    return math.hypot(point[0] - nearest[0], point[1] - nearest[1]), fraction


def bearingDegrees(start, end):
    deltaX = end[0] - start[0]
    deltaY = end[1] - start[1]
    return (math.degrees(math.atan2(deltaX, deltaY)) + 360.0) % 360.0


def angleDifference(firstAngle, secondAngle):
    difference = abs(firstAngle - secondAngle) % 360.0
    return min(difference, 360.0 - difference)


class SegmentGrid:
    def __init__(self, segments, cellSize=250.0):
        self.cellSize = cellSize
        self.cells = defaultdict(list)
        self.segments = segments
        for segmentIndex, segment in enumerate(segments):
            start, end = segment["start"], segment["end"]
            minX = math.floor(min(start[0], end[0]) / cellSize)
            maxX = math.floor(max(start[0], end[0]) / cellSize)
            minY = math.floor(min(start[1], end[1]) / cellSize)
            maxY = math.floor(max(start[1], end[1]) / cellSize)
            for cellX in range(minX, maxX + 1):
                for cellY in range(minY, maxY + 1):
                    self.cells[(cellX, cellY)].append(segmentIndex)

    def nearby(self, point, radius):
        minX = math.floor((point[0] - radius) / self.cellSize)
        maxX = math.floor((point[0] + radius) / self.cellSize)
        minY = math.floor((point[1] - radius) / self.cellSize)
        maxY = math.floor((point[1] + radius) / self.cellSize)
        found = set()
        for cellX in range(minX, maxX + 1):
            for cellY in range(minY, maxY + 1):
                found.update(self.cells.get((cellX, cellY), []))
        return found


def loadTrip(tripFolder):
    tripId = tripFolder.name
    trackPath = tripFolder / f"{tripId}.track"
    nodesPath = tripFolder / f"{tripId}.nodes"
    arcsPath = tripFolder / f"{tripId}.arcs"
    routePath = tripFolder / f"{tripId}.route"
    missingFiles = [str(path.name) for path in [trackPath, nodesPath, arcsPath, routePath] if not path.exists()]
    if missingFiles:
        raise FileNotFoundError(f"Trip {tripId} is missing files: {', '.join(missingFiles)}")

    trackRows = readRows(trackPath)
    nodeRows = readRows(nodesPath)
    arcRows = readRows(arcsPath)
    routeRows = readRows(routePath)

    if any(len(row) < 3 for row in trackRows):
        raise ValueError(f"{tripId}: every track row must contain longitude, latitude and timestamp")
    if any(len(row) < 2 for row in nodeRows) or any(len(row) < 2 for row in arcRows):
        raise ValueError(f"{tripId}: nodes and arcs must contain at least two columns")

    track = [(row[0], row[1], row[2]) for row in trackRows]
    nodes = [(row[0], row[1]) for row in nodeRows]
    arcs = [(int(row[0]), int(row[1])) for row in arcRows]
    route = [int(row[0]) for row in routeRows]

    invalidGps = sum(not (-180 <= lon <= 180 and -90 <= lat <= 90) for lon, lat, _ in track)
    invalidArcEndpoints = sum(source < 0 or target < 0 or source >= len(nodes) or target >= len(nodes) for source, target in arcs)
    duplicateArcs = len(arcs) - len(set(arcs))
    selfLoops = sum(source == target for source, target in arcs)
    invalidRouteIds = sum(arcId < 0 or arcId >= len(arcs) for arcId in route)
    return tripId, track, nodes, arcs, route, {
        "gpsCount": len(track), "nodeCount": len(nodes), "arcCount": len(arcs),
        "routeEntryCount": len(route), "invalidGpsCount": invalidGps,
        "invalidArcEndpointCount": invalidArcEndpoints, "duplicateArcCount": duplicateArcs,
        "selfLoopCount": selfLoops, "invalidRouteIdCount": invalidRouteIds,
        "routeEntriesAreArcIndexes": bool(route) and min(route) >= 0 and max(route) < len(arcs)
    }


def buildSegments(nodes, arcs, center):
    projectedNodes = [projectPoint(node, center) for node in nodes]
    segments = []
    for arcId, (source, target) in enumerate(arcs):
        if source < 0 or target < 0 or source >= len(nodes) or target >= len(nodes):
            continue
        start = projectedNodes[source]
        end = projectedNodes[target]
        segments.append({
            "arcId": arcId, "sourceNodeId": source, "targetNodeId": target,
            "start": start, "end": end, "bearing": bearingDegrees(start, end)
        })
    return segments


def alignReference(trackPoints, routeArcIds, segmentByArc, maxAlignmentDistance):
    """Greedy monotonic nearest-route alignment; distant or unavailable matches remain unmatched."""
    labels = []
    previousRoutePosition = 0
    for point in trackPoints:
        best = None
        for routePosition in range(previousRoutePosition, len(routeArcIds)):
            arcId = routeArcIds[routePosition]
            segment = segmentByArc.get(arcId)
            if segment is None:
                continue
            distance, fraction = pointSegmentDistance(point, segment["start"], segment["end"])
            if best is None or distance < best[0]:
                best = (distance, routePosition, arcId, fraction)
        if best is None or best[0] > maxAlignmentDistance:
            labels.append(None)
        else:
            labels.append({"distance": best[0], "routePosition": best[1], "referenceArcId": best[2], "fraction": best[3]})
            previousRoutePosition = best[1]
    return labels


def writeCsv(filePath, rows, fieldNames):
    filePath.parent.mkdir(parents=True, exist_ok=True)
    with filePath.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldNames)
        writer.writeheader()
        writer.writerows(rows)


def processTrip(tripFolder, partitionName, outputRoot, searchRadius, maxCandidates, maxAlignmentDistance):
    started = time.time()
    tripId, track, nodes, arcs, route, quality = loadTrip(tripFolder)
    validTrack = [(lon, lat, timestamp) for lon, lat, timestamp in track if -180 <= lon <= 180 and -90 <= lat <= 90]
    center = makeProjection([(lon, lat) for lon, lat, _ in validTrack] + nodes)
    projectedTrack = [projectPoint((lon, lat), center) if -180 <= lon <= 180 and -90 <= lat <= 90 else None for lon, lat, _ in track]
    segments = buildSegments(nodes, arcs, center)
    segmentByArc = {segment["arcId"]: segment for segment in segments}
    routeIsValid = quality["routeEntriesAreArcIndexes"] and all(arcId in segmentByArc for arcId in route)
    routeSegments = [segmentByArc[arcId] for arcId in route if arcId in segmentByArc] if routeIsValid else []
    routeConnectivityBreaks = sum(routeSegments[index - 1]["targetNodeId"] != routeSegments[index]["sourceNodeId"] for index in range(1, len(routeSegments)))
    labels = alignReference(projectedTrack, route, segmentByArc, maxAlignmentDistance) if routeIsValid else [None] * len(track)

    grid = SegmentGrid(segments)
    candidateRows = []
    observationRows = []
    candidateMisses = 0
    alignedCount = sum(label is not None for label in labels)
    previousBearing = None
    previousTime = None
    previousPoint = None

    for observationIndex, (rawPoint, projectedPoint, label) in enumerate(zip(track, projectedTrack, labels)):
        longitude, latitude, timestamp = rawPoint
        speed = None
        timeGap = None
        vehicleHeading = None
        if previousPoint is not None and projectedPoint is not None:
            timeGap = timestamp - previousTime
            distanceTravelled = math.dist(projectedPoint, previousPoint)
            if timeGap > 0:
                speed = distanceTravelled / timeGap
            if distanceTravelled > 0.5:
                vehicleHeading = bearingDegrees(previousPoint, projectedPoint)

        candidates = []
        if projectedPoint is not None:
            for segmentIndex in grid.nearby(projectedPoint, searchRadius):
                segment = segments[segmentIndex]
                distance, fraction = pointSegmentDistance(projectedPoint, segment["start"], segment["end"])
                if distance <= searchRadius:
                    candidates.append((distance, segment, fraction))
        candidates.sort(key=lambda item: item[0])
        candidates = candidates[:maxCandidates]

        candidateArcIds = {item[1]["arcId"] for item in candidates}
        if label is not None and label["referenceArcId"] not in candidateArcIds:
            candidateMisses += 1

        for rank, (distance, segment, fraction) in enumerate(candidates, start=1):
            headingDifference = angleDifference(vehicleHeading, segment["bearing"]) if vehicleHeading is not None else None
            candidateRows.append({
                "tripId": tripId, "partition": partitionName, "observationIndex": observationIndex,
                "candidateRank": rank, "candidateArcId": segment["arcId"],
                "sourceNodeId": segment["sourceNodeId"], "targetNodeId": segment["targetNodeId"],
                "distanceMeters": round(distance, 4), "segmentFraction": round(fraction, 6),
                "headingDifferenceDegrees": round(headingDifference, 4) if headingDifference is not None else "",
                "vehicleSpeedMetersPerSecond": round(speed, 4) if speed is not None else "",
                "timeGapSeconds": round(timeGap, 4) if timeGap is not None else ""
            })

        observationRows.append({
            "tripId": tripId, "partition": partitionName, "observationIndex": observationIndex,
            "longitude": longitude, "latitude": latitude, "timestamp": timestamp,
            "referenceArcId": label["referenceArcId"] if label else "",
            "alignmentDistanceMeters": round(label["distance"], 4) if label else "",
            "alignmentStatus": "aligned" if label else "unmatched",
            "candidateCount": len(candidates)
        })

        if projectedPoint is not None:
            previousPoint = projectedPoint
            previousTime = timestamp
            if vehicleHeading is not None:
                previousBearing = vehicleHeading

    tripOutput = outputRoot / partitionName / tripId
    candidateFields = ["tripId", "partition", "observationIndex", "candidateRank", "candidateArcId",
                       "sourceNodeId", "targetNodeId", "distanceMeters", "segmentFraction",
                       "headingDifferenceDegrees", "vehicleSpeedMetersPerSecond", "timeGapSeconds"]
    observationFields = ["tripId", "partition", "observationIndex", "longitude", "latitude", "timestamp",
                         "referenceArcId", "alignmentDistanceMeters", "alignmentStatus", "candidateCount"]
    writeCsv(tripOutput / "candidateRecords.csv", candidateRows, candidateFields)
    writeCsv(tripOutput / "observationLabels.csv", observationRows, observationFields)

    validTimes = [track[index][2] for index in range(len(track))]
    timeGaps = [validTimes[index] - validTimes[index - 1] for index in range(1, len(validTimes))]
    validGaps = [gap for gap in timeGaps if gap > 0]
    quality.update({
        "partition": partitionName, "tripId": tripId, "validRoadSegmentCount": len(segments),
        "routeIsValidArcSequence": routeIsValid, "routeConnectivityBreakCount": routeConnectivityBreaks,
        "alignedObservationCount": alignedCount,
        "unmatchedObservationCount": len(track) - alignedCount,
        "candidateRecordCount": len(candidateRows),
        "observationsWithNoCandidates": sum(row["candidateCount"] == 0 for row in observationRows),
        "candidateMissCount": candidateMisses,
        "candidateMissRate": candidateMisses / alignedCount if alignedCount else None,
        "averageCandidatesPerObservation": len(candidateRows) / len(track) if track else 0,
        "medianPositiveTimeGapSeconds": statistics.median(validGaps) if validGaps else None,
        "processingSeconds": round(time.time() - started, 3)
    })
    with (tripOutput / "tripReport.json").open("w", encoding="utf-8") as file:
        json.dump(quality, file, indent=2)
    return quality


def main():
    parser = argparse.ArgumentParser(description="PathSnap Member 1 dataset preprocessing")
    parser.add_argument("--dataDir", default="datasets", help="Folder containing train and student_test")
    parser.add_argument("--outputDir", default="memberOneOutput", help="Where processed outputs are written")
    parser.add_argument("--searchRadius", type=float, default=100.0, help="Candidate search radius in metres")
    parser.add_argument("--maxCandidates", type=int, default=8, help="Maximum candidates per GPS observation")
    parser.add_argument("--maxAlignmentDistance", type=float, default=200.0,
                        help="Maximum GPS-to-reference distance in metres for accepting an alignment")
    parser.add_argument("--partitions", nargs="+", default=["train", "student_test"], choices=["train", "student_test"])
    args = parser.parse_args()

    dataRoot = Path(args.dataDir)
    outputRoot = Path(args.outputDir)
    if args.searchRadius <= 0 or args.maxCandidates <= 0 or args.maxAlignmentDistance <= 0:
        parser.error("--searchRadius, --maxCandidates and --maxAlignmentDistance must be positive")
    allReports = []
    for partitionName in args.partitions:
        partitionFolder = dataRoot / partitionName
        if not partitionFolder.is_dir():
            print(f"Skipping missing partition: {partitionFolder}")
            continue
        tripFolders = sorted(folder for folder in partitionFolder.iterdir() if folder.is_dir())
        print(f"Processing {len(tripFolders)} trips in {partitionName}...")
        for tripFolder in tripFolders:
            try:
                report = processTrip(tripFolder, partitionName, outputRoot, args.searchRadius, args.maxCandidates,
                                     args.maxAlignmentDistance)
                allReports.append(report)
                print(f"  {report['tripId']}: {report['gpsCount']} observations, "
                      f"{report['candidateRecordCount']} candidates, "
                      f"{report['alignedObservationCount']} aligned")
            except Exception as error:
                failure = {"tripId": tripFolder.name, "partition": partitionName, "processingError": str(error)}
                allReports.append(failure)
                print(f"  ERROR {tripFolder.name}: {error}")

    outputRoot.mkdir(parents=True, exist_ok=True)
    with (outputRoot / "datasetReport.json").open("w", encoding="utf-8") as file:
        json.dump(allReports, file, indent=2)
    successful = sum("processingError" not in report for report in allReports)
    print(f"\nFinished: {successful}/{len(allReports)} trips processed.")
    print(f"Reports and CSV outputs: {outputRoot.resolve()}")
    if successful != len(allReports):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
