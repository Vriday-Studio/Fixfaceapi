import { useCallback } from "react";

export function useFrameOutput({
  focusIndexRef,
  focusTargetRef,
  sendGreenSnapshot,
  emitCrowdByGid,
  setTable,
  setTotals,
  gesturesOnRef,
  deviceId,
  sessionId,
  backend,
}) {
  const sameRow = (a, b) =>
    a.idx === b.idx &&
    a.gender === b.gender &&
    a.ageGroup === b.ageGroup &&
    a.zone === b.zone &&
    a.name === b.name &&
    a.gesture === b.gesture &&
    a.emotion === b.emotion &&
    a.distance === b.distance;

  const commitFrameOutput = useCallback(
    ({ cand, peopleForPost, rows, total, green, red, fresh }) => {
      if (cand.length > 0) {
        sendGreenSnapshot(cand);

        emitCrowdByGid({
          deviceId,
          sessionId: sessionId || "web-" + deviceId,
          timeISO: new Date().toISOString(),
          backend,
          totals: green,
          gesture: gesturesOnRef.current ? fresh : null,
          focusIndex: focusIndexRef.current,
          focusTarget: focusTargetRef.current,
          people: cand,
        });
      }

      while (rows.length < 5) {
        rows.push({
          idx: rows.length + 1,
          gender: "-",
          ageGroup: "-",
          zone: "-",
          name: "-",
          gesture: "-",
          emotion: "-",
          distance: "-",
        });
      }

      setTable((prev) => {
        const same =
          prev.length === rows.length &&
          prev.every((r, i) => sameRow(r, rows[i]));
        return same ? prev : rows;
      });

      setTotals((prev) =>
        prev.all === total && prev.green === green && prev.red === red
          ? prev
          : { all: total, green, red }
      );
    },
    [
      backend,
      deviceId,
      emitCrowdByGid,
      focusIndexRef,
      focusTargetRef,
      gesturesOnRef,
      sendGreenSnapshot,
      sessionId,
      setTable,
      setTotals,
    ]
  );

  const updateFocusTarget = useCallback(
    ({ focusIndex, cand, peopleForPost }) => {
      if (focusIndex >= 0) {
        const p = cand[focusIndex];
        focusIndexRef.current = peopleForPost.indexOf(p);
        focusTargetRef.current = { name: p.name || null, gid: p.gid || null };
        return;
      }

      focusIndexRef.current = -1;
      focusTargetRef.current = null;
    },
    [focusIndexRef, focusTargetRef]
  );

  return {
    commitFrameOutput,
    updateFocusTarget,
  };
}
