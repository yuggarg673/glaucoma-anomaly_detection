export default function Disclaimer({ text }) {
  return (
    <div
      role="note"
      className="rounded-xl border border-amber-300 bg-amber-50 px-5 py-4 text-[15px] text-amber-950"
    >
      <strong>Screening aid only.</strong>{" "}
      {text ||
        "This tool does not give a medical diagnosis. Results should be reviewed by an ophthalmologist."}
    </div>
  );
}
