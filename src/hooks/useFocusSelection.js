import { useCallback } from "react";

export function useFocusSelection({ wNear, wCenter, wMouth }) {
  const selectFocus = useCallback(
    (peopleForPost) => {
      const pool = peopleForPost.filter((p) => p.zone === "green");
      const cand = pool.length ? pool : peopleForPost;

      let focusIndex = cand.length ? 0 : -1;
      let focusScore = -1;

      for (let idx = 0; idx < cand.length; idx++) {
        const p = cand[idx];
        let sNear = 0;
        const z = p?.posCam?.z;
        if (Number.isFinite(z) && z > 0) {
          sNear = Math.max(
            0,
            Math.min(1, (2.0 - Math.min(2.0, Math.max(0.3, z))) / (2.0 - 0.3))
          );
        }
        const sCenter = 1 - Math.max(0, Math.min(1, p.centerNorm ?? 1));
        const sMouth = Math.max(0, Math.min(1, p.mouthActivity ?? 0));
        const score = wNear * sNear + wCenter * sCenter + wMouth * sMouth;
        if (score > focusScore) {
          focusScore = score;
          focusIndex = idx;
        }
      }

      return { cand, focusIndex };
    },
    [wCenter, wMouth, wNear]
  );

  return { selectFocus };
}
