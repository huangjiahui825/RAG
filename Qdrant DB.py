from qdrant_client import QdrantClient
from qdrant_client.models import PointIdsList

client = QdrantClient(host="localhost", port=6333)
collection_name = "rag_data"

print("[INFO] Scanning all data in collection:", collection_name)

# 显示所有 PDF 和 Excel 数据
for source_type in ["pdf", "excel"]:
    scroll_result = client.scroll(
        collection_name=collection_name,
        scroll_filter={
            "must": [
                {"key": "source", "match": {"value": source_type}}
            ]
        },
        with_payload=True,
        limit=10000
    )
    points = scroll_result[0]
    print(f"\n=== {source_type.upper()} Data ===")
    for point in points:
        print(f"ID: {point.id}, Payload: {point.payload}")

# === ✅ Sheet 级别删除 ===
target_sheet_name = "Sheet1"  # ✅ 你可以改成任意 Sheet 名称
sheet_ids_to_delete = []
scroll_result = client.scroll(
    collection_name=collection_name,
    scroll_filter={
        "must": [
            {"key": "source", "match": {"value": "excel"}}
        ]
    },
    with_payload=True,
    limit=10000
)
for point in scroll_result[0]:
    text = point.payload.get("text", "")
    if f"[Sheet: {target_sheet_name}]" in text:
        sheet_ids_to_delete.append(point.id)

if sheet_ids_to_delete:
    client.delete(
        collection_name=collection_name,
        points_selector=PointIdsList(points=sheet_ids_to_delete)
    )
    print(f"[INFO] Deleted {len(sheet_ids_to_delete)} vectors for Sheet '{target_sheet_name}'.")
else:
    print(f"[INFO] No vectors found for Sheet '{target_sheet_name}'.")

# === ✅ PDF 与 Excel 文件级删除（原逻辑保留） ===
for source_type in ["pdf", "excel"]:
    scroll_result = client.scroll(
        collection_name=collection_name,
        scroll_filter={
            "must": [
                {"key": "source", "match": {"value": source_type}}
            ]
        },
        with_payload=False,
        limit=10000
    )
    ids_to_delete = [point.id for point in scroll_result[0]]
    if ids_to_delete:
        client.delete(
            collection_name=collection_name,
            points_selector=PointIdsList(points=ids_to_delete)
        )
        print(f"[INFO] Deleted {len(ids_to_delete)} vectors of type '{source_type}'.")

print("[INFO] All PDF, Excel file-level, and specified Sheet vectors processed.")
