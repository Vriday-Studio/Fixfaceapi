import React from "react";
import PeopleTable from "./PeopleTable";

export default function GuestTablePanel({ table }) {
  return (
    <div className="panel tablewrap" style={{ padding: 12 }}>
      <PeopleTable table={table} />
    </div>
  );
}
