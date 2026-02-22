import { Check, Send } from "lucide-react";
import { Button } from "@/components/ui/button";

interface CorrectionToolbarProps {
  hasChanges: boolean;
  onConfirm: () => void;
  onSubmitCorrection: () => void;
  isSubmitting?: boolean;
}

export function CorrectionToolbar({
  hasChanges,
  onConfirm,
  onSubmitCorrection,
  isSubmitting,
}: CorrectionToolbarProps) {
  return (
    <div className="flex items-center gap-2 pt-2">
      {hasChanges ? (
        <Button onClick={onSubmitCorrection} disabled={isSubmitting}>
          <Send className="h-4 w-4 mr-2" />
          {isSubmitting ? "Submitting..." : "Submit Correction"}
        </Button>
      ) : (
        <Button variant="outline" onClick={onConfirm} disabled={isSubmitting}>
          <Check className="h-4 w-4 mr-2" />
          Looks Correct
        </Button>
      )}
    </div>
  );
}
