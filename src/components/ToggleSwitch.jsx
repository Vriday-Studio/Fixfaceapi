export default function ToggleSwitch({ checked, onChange, label }) {
  return (
    <label
      style={{
        display: "inline-flex",
        alignItems: "center",
        gap: 8,
        cursor: "pointer",
        fontSize: 13,
        color: "#ebebeb",
      }}
    >
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        style={{ accentColor: "#0ea5e9", width: 18, height: 18 }}
      />
      {label}
    </label>
  );
}
