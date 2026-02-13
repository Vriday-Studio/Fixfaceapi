import React from "react";

export default function PeopleTable({ table }) {
  return (
    <table className="table">
      <thead>
        <tr>
          {[
            "#",
            "Name",
            "Gesture",
            "Emotion",
            "Zone",
            "AgeGroup",
            "Gender",
            "Distance",
          ].map((h) => (
            <th key={h}>{h}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {table.map((r) => (
          <tr key={r.idx}>
            <td>{r.idx}</td>
            <td>{r.name}</td>
            <td>{r.gesture ?? "-"}</td>
            <td>{r.emotion ?? "-"}</td>
            <td
              className={
                r.zone === "green"
                  ? "zone-green"
                  : r.zone === "red"
                    ? "zone-red"
                    : "zone-unk"
              }
            >
              {r.zone}
            </td>
            <td>{r.ageGroup}</td>
            <td>{r.gender}</td>
            <td>{r.distance}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}
