import { useCallback, useState, type DragEvent } from "react";
import { Upload } from "lucide-react";
import { useUploadFile } from "@/api/client";
import { toast } from "sonner";

interface FileUploadProps {
  accept?: string;
  prompt?: string;
}

export function FileUpload({
  accept = ".txt,.csv,.json,.jsonl",
  prompt = "Drag and drop a file, or browse",
}: FileUploadProps) {
  const upload = useUploadFile();
  const [dragging, setDragging] = useState(false);

  const hasFiles = (event: DragEvent<HTMLDivElement>) =>
    Array.from(event.dataTransfer.types).includes("Files");

  const handleFile = useCallback(
    (file: File) => {
      upload.mutate(file, {
        onSuccess: (data: { count: number; message: string }) => {
          toast.success(data.message);
        },
        onError: (err: Error) => {
          toast.error(err.message);
        },
      });
    },
    [upload]
  );

  return (
    <div
      className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
        dragging ? "border-primary bg-primary/5" : "border-muted-foreground/25"
      }`}
      onDragOver={(e) => {
        if (!hasFiles(e)) {
          return;
        }
        e.preventDefault();
        setDragging(true);
      }}
      onDragLeave={(e) => {
        if (!hasFiles(e)) {
          return;
        }
        setDragging(false);
      }}
      onDrop={(e) => {
        if (!hasFiles(e)) {
          return;
        }
        e.preventDefault();
        setDragging(false);
        const file = e.dataTransfer.files[0];
        if (file) handleFile(file);
      }}
    >
      <Upload className="h-8 w-8 mx-auto mb-2 text-muted-foreground" />
      <p className="text-sm text-muted-foreground mb-2">
        {prompt},{" "}
        <label className="text-primary cursor-pointer underline">
          browse
          <input
            type="file"
            className="hidden"
            accept={accept}
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) handleFile(file);
            }}
          />
        </label>
      </p>
      {upload.isPending && (
        <p className="text-xs text-muted-foreground">Uploading...</p>
      )}
    </div>
  );
}
